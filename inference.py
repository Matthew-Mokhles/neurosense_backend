"""
ASD Detection - Inference Script
================================================
Pipeline:
    Video → ROMP (backend) → 3D Skeleton → ASD Model → JSON Report

Usage:
    # من skeleton جاهز (.npz)
    python inference.py --skeleton path/to/skeleton.npz --activity arm_swing

    # من فولدر فيه أكتر من activity
    python inference.py --session_dir path/to/session_folder/

    # من الكود مباشرة
    from inference import ASDInference
    engine = ASDInference(model_path='checkpoints/best_model.pth')
    result = engine.predict_from_skeleton(skeleton_array, activity='arm_swing')
"""

import torch
import numpy as np
import json
import os
import argparse
from pathlib import Path
from datetime import datetime


# ============================================================
# ACTIVITY MAPPING — نفس الـ 11 activities في الـ MMASD
# ============================================================
ACTIVITY_MAP = {
    'arm_swing':            {'id': 0,  'display': 'Arm Swing',             'theme': 'Robotic'},
    'body_swing':           {'id': 1,  'display': 'Body Swing',            'theme': 'Robotic'},
    'chest_expansion':      {'id': 2,  'display': 'Chest Expansion',       'theme': 'Robotic'},
    'squat':                {'id': 3,  'display': 'Squat',                 'theme': 'Robotic'},
    'drumming':             {'id': 4,  'display': 'Drumming',              'theme': 'Music'},
    'maracas_forward':      {'id': 5,  'display': 'Maracas Forward Shaking','theme': 'Music'},
    'maracas_shaking':      {'id': 6,  'display': 'Maracas Shaking',       'theme': 'Music'},
    'sing_and_clap':        {'id': 7,  'display': 'Sing and Clap',         'theme': 'Music'},
    'frog_pose':            {'id': 8,  'display': 'Frog Pose',             'theme': 'Yoga'},
    'tree_pose':            {'id': 9,  'display': 'Tree Pose',             'theme': 'Yoga'},
    'twist_pose':           {'id': 10, 'display': 'Twist Pose',            'theme': 'Yoga'},
}

# Symptom labels — نفس ترتيب الـ model
SYMPTOM_LABELS = [
    'hand_flapping',
    'head_banging',
    'body_spinning',
    'hand_stereotypy',
    'body_rocking',
    'motor_coordination',
    'rhythmic_movement',
    'balance_coordination',
    'social_interaction',
    'general_behavior',
    'typically_developing',
]

SYMPTOM_DISPLAY = {
    'hand_flapping':       'Hand Flapping',
    'head_banging':        'Head Banging',
    'body_spinning':       'Body Spinning',
    'hand_stereotypy':     'Hand Stereotypy',
    'body_rocking':        'Body Rocking',
    'motor_coordination':  'Motor Coordination Issues',
    'rhythmic_movement':   'Rhythmic Movement',
    'balance_coordination':'Balance & Coordination Issues',
    'social_interaction':  'Social Interaction Difficulty',
    'general_behavior':    'General ASD Behavior',
    'typically_developing':'Typically Developing',
}


# ============================================================
# PREPROCESSING — نفس منطق الـ data_loader
# ============================================================
class SkeletonPreprocessor:
    """يحول أي skeleton input للـ format الصح للموديل"""

    def __init__(self, max_frames=150, num_joints=24):
        self.max_frames = max_frames
        self.num_joints = num_joints

    def process(self, skeleton: np.ndarray) -> np.ndarray:
        """
        Input:  أي shape من skeleton data
        Output: [150, 24, 3] normalized
        """
        skeleton = self._fix_shape(skeleton)
        skeleton = self._resample(skeleton)
        skeleton = self._normalize(skeleton)
        return skeleton.astype(np.float32)

    def _fix_shape(self, skeleton: np.ndarray) -> np.ndarray:
        """يضبط الـ shape بغض النظر عن الـ input format"""

        # لو (frames, joints, coords) — ده الـ format الصح
        if skeleton.ndim == 3:
            frames, joints, coords = skeleton.shape

            # لو عدد الـ joints أكتر من 24 خد أول 24
            if joints > self.num_joints:
                skeleton = skeleton[:, :self.num_joints, :]
            # لو أقل من 24 زود بـ zeros
            elif joints < self.num_joints:
                pad = np.zeros((frames, self.num_joints - joints, coords))
                skeleton = np.concatenate([skeleton, pad], axis=1)

            # لو coords أكتر من 3 خد أول 3
            if coords > 3:
                skeleton = skeleton[:, :, :3]

        # لو (joints, coords) — frame واحدة بس
        elif skeleton.ndim == 2:
            skeleton = skeleton[np.newaxis, :, :]  # → (1, joints, coords)
            skeleton = self._fix_shape(skeleton)

        # لو (batch, frames, joints, coords) — خد أول sample
        elif skeleton.ndim == 4:
            skeleton = skeleton[0]
            skeleton = self._fix_shape(skeleton)

        else:
            raise ValueError(f"❌ Skeleton shape غير متوقع: {skeleton.shape}")

        return skeleton

    def _resample(self, skeleton: np.ndarray) -> np.ndarray:
        """يعمل resample للـ skeleton لـ 150 frames بالظبط"""
        current_frames = skeleton.shape[0]

        if current_frames == 0:
            return np.zeros((self.max_frames, self.num_joints, 3))

        if current_frames == self.max_frames:
            return skeleton

        if current_frames > self.max_frames:
            # Downsample
            indices = np.linspace(0, current_frames - 1, self.max_frames, dtype=int)
            return skeleton[indices]
        else:
            # Upsample بـ interpolation
            indices = np.linspace(0, current_frames - 1, self.max_frames)
            resampled = np.zeros((self.max_frames, self.num_joints, 3))
            for i, idx in enumerate(indices):
                low = int(idx)
                high = min(low + 1, current_frames - 1)
                alpha = idx - low
                resampled[i] = (1 - alpha) * skeleton[low] + alpha * skeleton[high]
            return resampled

    def _normalize(self, skeleton: np.ndarray) -> np.ndarray:
        """يعمل normalize للـ skeleton coordinates لـ [-1, 1]"""
        non_zero_mask = np.any(skeleton != 0, axis=(1, 2))

        if non_zero_mask.sum() == 0:
            return skeleton

        non_zero_frames = skeleton[non_zero_mask].copy()

        # Center
        center = np.mean(non_zero_frames, axis=(0, 1), keepdims=True)
        non_zero_frames -= center

        # Scale
        max_val = np.abs(non_zero_frames).max()
        if max_val > 0:
            non_zero_frames /= max_val

        skeleton[non_zero_mask] = non_zero_frames
        return skeleton


# ============================================================
# SKELETON LOADER — يقرأ الـ .npz files
# ============================================================
class SkeletonLoader:
    """يقرأ skeleton من أي format"""

    @staticmethod
    def from_npz_file(path: str) -> np.ndarray:
        """يقرأ skeleton من ملف .npz واحد"""
        data = np.load(path, allow_pickle=True)

        # جرب الـ keys المختلفة
        for key in ['pred_j3d', 'joints', 'skeleton', 'keypoints_3d']:
            if key in data:
                return data[key]

        # لو مفيش key معروف خد أول array
        keys = [k for k in data.keys() if not k.startswith('_')]
        if keys:
            return data[keys[0]]

        raise ValueError(f"❌ مش لاقي skeleton data في: {path}")

    @staticmethod
    def from_npz_folder(folder_path: str) -> np.ndarray:
        """يقرأ skeleton من فولدر فيه .npz files (كل ملف = frame)"""
        folder = Path(folder_path)
        files = sorted(folder.glob('*.npz'))

        if not files:
            raise ValueError(f"❌ مش لاقي .npz files في: {folder_path}")

        frames = []
        for f in files:
            try:
                data = np.load(f, allow_pickle=True)
                for key in ['pred_j3d', 'joints', 'skeleton']:
                    if key in data:
                        frame = data[key]
                        if frame.ndim == 3:   # (batch, joints, coords)
                            frame = frame[0]
                        frames.append(frame)
                        break
            except Exception:
                continue

        if not frames:
            raise ValueError(f"❌ مش قدر يقرأ أي frame من: {folder_path}")

        return np.array(frames)  # (frames, joints, coords)

    @staticmethod
    def from_numpy(array: np.ndarray) -> np.ndarray:
        """يقبل numpy array مباشرة"""
        return array


# ============================================================
# REPORT GENERATOR
# ============================================================
class ReportGenerator:
    """يولد التقرير النهائي"""

    # حدود الـ ASD probability
    RISK_THRESHOLDS = {
        'low':      (0.0,  0.35),
        'moderate': (0.35, 0.65),
        'high':     (0.65, 1.0),
    }

    @classmethod
    def generate(cls,
                 asd_prob: float,
                 symptom_probs: np.ndarray,
                 severity: float,
                 activity: str,
                 child_info: dict = None) -> dict:

        risk_level = cls._get_risk_level(asd_prob)
        top_symptoms = cls._get_top_symptoms(symptom_probs)
        severity_norm = cls._normalize_severity(severity)

        report = {
            'timestamp': datetime.now().isoformat(),
            'activity': {
                'key': activity,
                'display': ACTIVITY_MAP.get(activity, {}).get('display', activity),
                'theme':   ACTIVITY_MAP.get(activity, {}).get('theme', 'Unknown'),
            },
            'asd_screening': {
                'probability':   round(float(asd_prob), 4),
                'probability_pct': round(float(asd_prob) * 100, 1),
                'risk_level':    risk_level,
                'risk_label':    cls._risk_label(risk_level),
            },
            'severity': {
                'raw_score':    round(float(severity), 4),
                'normalized':   round(float(severity_norm), 2),
                'scale':        '0-10 (ADOS comparison score range)',
            },
            'symptoms': {
                'top_symptom':     top_symptoms[0]['name'] if top_symptoms else 'N/A',
                'top_symptom_display': top_symptoms[0]['display'] if top_symptoms else 'N/A',
                'all_symptoms':    top_symptoms,
            },
            'disclaimer': (
                "⚠️ هذه النتائج للفحص المبكر فقط وليست تشخيصاً طبياً. "
                "يُرجى استشارة متخصص مؤهل."
            ),
        }

        if child_info:
            report['child_info'] = child_info

        return report

    @classmethod
    def _get_risk_level(cls, prob: float) -> str:
        for level, (low, high) in cls.RISK_THRESHOLDS.items():
            if low <= prob < high:
                return level
        return 'high'

    @staticmethod
    def _risk_label(level: str) -> str:
        return {
            'low':      '🟢 احتمال منخفض',
            'moderate': '🟡 احتمال متوسط',
            'high':     '🔴 احتمال مرتفع',
        }.get(level, '⚪ غير محدد')

    @staticmethod
    def _get_top_symptoms(probs: np.ndarray, top_k: int = 3) -> list:
        top_indices = np.argsort(probs)[::-1][:top_k]
        results = []
        for idx in top_indices:
            if idx < len(SYMPTOM_LABELS):
                name = SYMPTOM_LABELS[idx]
                results.append({
                    'name':        name,
                    'display':     SYMPTOM_DISPLAY.get(name, name),
                    'probability': round(float(probs[idx]), 4),
                    'probability_pct': round(float(probs[idx]) * 100, 1),
                })
        return results

    @staticmethod
    def _normalize_severity(raw: float, min_val: float = -3.0, max_val: float = 3.0) -> float:
        """يحول الـ raw severity score لـ scale من 0 إلى 10"""
        normalized = (raw - min_val) / (max_val - min_val)
        return round(float(np.clip(normalized * 10, 0, 10)), 2)


# ============================================================
# ASD INFERENCE ENGINE
# ============================================================

def _patch_numpy2_checkpoint_compat():
    """Map numpy._core so NumPy 1.x can load checkpoints saved with NumPy 2.x."""
    import sys
    if 'numpy._core' in sys.modules:
        return
    import numpy as np
    if hasattr(np, '_core'):
        return
    core = np.core
    sys.modules['numpy._core'] = core
    sys.modules['numpy._core.multiarray'] = core.multiarray
    sys.modules['numpy._core.umath'] = core.umath


class ASDInference:
    """
    الـ Engine الرئيسي للـ inference

    مثال:
        engine = ASDInference(model_path='checkpoints/best_model.pth')

        # من numpy array
        result = engine.predict_from_skeleton(skeleton_array, activity='arm_swing')

        # من ملف .npz
        result = engine.predict_from_file('skeleton.npz', activity='frog_pose')

        # من فولدر فيه .npz frames
        result = engine.predict_from_folder('skeleton_folder/', activity='tree_pose')
    """

    def __init__(self,
                 model_path: str,
                 num_symptoms: int = 11,
                 num_joints: int = 24,
                 device: str = None):

        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.preprocessor = SkeletonPreprocessor()
        self.report_gen = ReportGenerator()

        print(f"🚀 ASD Inference Engine")
        print(f"   Device: {self.device}")
        print(f"   Model: {model_path}")

        self.model = self._load_model(model_path, num_symptoms, num_joints)

    def _load_model(self, path, num_symptoms, num_joints):
        """يحمّل الموديل من الـ checkpoint"""
        # Import هنا عشان يشتغل من أي مكان
        import sys
        sys.path.append(str(Path(__file__).parent))
        from asd_model import ASD_Detection_Model

        model = ASD_Detection_Model(
            num_symptoms=num_symptoms,
            num_joints=num_joints,
            in_channels=3
        )

        _patch_numpy2_checkpoint_compat()
        checkpoint = torch.load(str(path), map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()

        print(f"   ✅ Model loaded (epoch {checkpoint.get('epoch', '?')})")
        return model

    def predict_from_skeleton(self,
                               skeleton: np.ndarray,
                               activity: str = 'general_behavior',
                               child_info: dict = None) -> dict:
        """
        يعمل prediction من numpy array مباشرة

        Args:
            skeleton:   numpy array أي shape
            activity:   اسم الـ activity (من ACTIVITY_MAP)
            child_info: معلومات الطفل (اختياري) {'name': ..., 'age': ...}

        Returns:
            dict: التقرير الكامل
        """
        if activity not in ACTIVITY_MAP:
            print(f"⚠️  Activity '{activity}' مش موجودة، هستخدم 'general_behavior'")

        # Preprocessing
        processed = self.preprocessor.process(skeleton)

        # إلى tensor
        tensor = torch.FloatTensor(processed).unsqueeze(0).to(self.device)
        # shape: [1, 150, 24, 3]

        # Inference
        with torch.no_grad():
            outputs = self.model(tensor)

        # استخراج النتايج
        asd_prob = outputs['asd_probability'].squeeze().cpu().item()
        symptom_logits = outputs['symptom_logits'].squeeze().cpu().numpy()
        severity = outputs['severity'].squeeze().cpu().item()

        # Softmax على الـ symptom logits
        symptom_probs = self._softmax(symptom_logits)

        # توليد التقرير
        report = ReportGenerator.generate(
            asd_prob=asd_prob,
            symptom_probs=symptom_probs,
            severity=severity,
            activity=activity,
            child_info=child_info,
        )

        return report

    def predict_from_file(self,
                           npz_path: str,
                           activity: str = 'general_behavior',
                           child_info: dict = None) -> dict:
        """يعمل prediction من ملف .npz واحد"""
        skeleton = SkeletonLoader.from_npz_file(npz_path)
        return self.predict_from_skeleton(skeleton, activity, child_info)

    def predict_from_folder(self,
                             folder_path: str,
                             activity: str = 'general_behavior',
                             child_info: dict = None) -> dict:
        """يعمل prediction من فولدر فيه .npz frames"""
        skeleton = SkeletonLoader.from_npz_folder(folder_path)
        return self.predict_from_skeleton(skeleton, activity, child_info)

    def predict_full_session(self,
                              activities_dict: dict,
                              child_info: dict = None) -> dict:
        """
        يعمل prediction لـ session كاملة فيها أكتر من activity

        Args:
            activities_dict: {
                'arm_swing': skeleton_array أو path,
                'frog_pose': skeleton_array أو path,
                ...
            }
            child_info: معلومات الطفل

        Returns:
            dict: تقرير شامل للـ session كلها
        """
        activity_results = []
        asd_probs = []

        for activity, skeleton_input in activities_dict.items():
            print(f"   🔄 Processing: {activity}")

            try:
                if isinstance(skeleton_input, np.ndarray):
                    result = self.predict_from_skeleton(skeleton_input, activity, child_info)
                elif isinstance(skeleton_input, str):
                    path = Path(skeleton_input)
                    if path.is_dir():
                        result = self.predict_from_folder(str(path), activity, child_info)
                    else:
                        result = self.predict_from_file(str(path), activity, child_info)
                else:
                    print(f"   ⚠️  Input type غير متوقع لـ {activity}")
                    continue

                activity_results.append(result)
                asd_probs.append(result['asd_screening']['probability'])

            except Exception as e:
                print(f"   ❌ Error في {activity}: {e}")
                continue

        if not activity_results:
            return {'error': 'مفيش نتايج — تأكد من الـ input'}

        # حساب المتوسط لكل الـ activities
        avg_prob = float(np.mean(asd_probs))
        max_prob = float(np.max(asd_probs))

        session_report = {
            'timestamp': datetime.now().isoformat(),
            'child_info': child_info or {},
            'session_summary': {
                'activities_completed':  len(activity_results),
                'activities_total':      11,
                'avg_asd_probability':   round(avg_prob, 4),
                'avg_probability_pct':   round(avg_prob * 100, 1),
                'max_asd_probability':   round(max_prob, 4),
                'overall_risk_level':    ReportGenerator._get_risk_level(avg_prob),
                'overall_risk_label':    ReportGenerator._risk_label(
                                            ReportGenerator._get_risk_level(avg_prob)),
            },
            'per_activity_results': activity_results,
            'disclaimer': (
                "⚠️ هذه النتائج للفحص المبكر فقط وليست تشخيصاً طبياً. "
                "يُرجى استشارة متخصص مؤهل."
            ),
        }

        return session_report

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()


# ============================================================
# CLI — تشغيل من الـ terminal
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description='ASD Detection Inference')
    parser.add_argument('--model',      type=str, default=r'D:\ASD_Detection\checkpoints\best_model.pth')
    parser.add_argument('--skeleton',   type=str, help='Path to .npz skeleton file')
    parser.add_argument('--folder',     type=str, help='Path to folder with .npz frames')
    parser.add_argument('--session_dir',type=str, help='Path to session folder (subfolders = activities)')
    parser.add_argument('--activity',   type=str, default='arm_swing',
                        choices=list(ACTIVITY_MAP.keys()),
                        help='Activity name')
    parser.add_argument('--output',     type=str, default=None, help='Save report to JSON file')
    parser.add_argument('--child_name', type=str, default=None)
    parser.add_argument('--child_age',  type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    # معلومات الطفل (اختياري)
    child_info = None
    if args.child_name or args.child_age:
        child_info = {'name': args.child_name, 'age': args.child_age}

    # تشغيل الـ engine
    engine = ASDInference(model_path=args.model)

    print(f"\n{'='*60}")

    # اختيار نوع الـ input
    if args.session_dir:
        # Session كاملة — كل subfolder هو activity
        session_path = Path(args.session_dir)
        activities = {}
        for subfolder in session_path.iterdir():
            if subfolder.is_dir() and subfolder.name in ACTIVITY_MAP:
                activities[subfolder.name] = str(subfolder)
        print(f"📁 Session mode: {len(activities)} activities found")
        report = engine.predict_full_session(activities, child_info)

    elif args.folder:
        print(f"📁 Folder mode: {args.folder}")
        report = engine.predict_from_folder(args.folder, args.activity, child_info)

    elif args.skeleton:
        print(f"📄 File mode: {args.skeleton}")
        report = engine.predict_from_file(args.skeleton, args.activity, child_info)

    else:
        # Demo بـ dummy data
        print("🎮 Demo mode (dummy skeleton data)")
        dummy_skeleton = np.random.randn(150, 24, 3).astype(np.float32)
        report = engine.predict_from_skeleton(dummy_skeleton, 'arm_swing',
                                              {'name': 'Test Child', 'age': 7})

    # عرض النتيجة
    print(f"\n{'='*60}")
    print("📊 SCREENING REPORT")
    print(f"{'='*60}")

    if 'session_summary' in report:
        # Session report
        s = report['session_summary']
        print(f"\n👶 Child: {report.get('child_info', {}).get('name', 'N/A')}")
        print(f"✅ Activities completed: {s['activities_completed']}/{s['activities_total']}")
        print(f"📈 Average ASD Probability: {s['avg_probability_pct']}%")
        print(f"🎯 Overall Risk: {s['overall_risk_label']}")
    else:
        # Single activity report
        a = report['asd_screening']
        print(f"\n🏃 Activity: {report['activity']['display']}")
        print(f"📈 ASD Probability: {a['probability_pct']}%")
        print(f"🎯 Risk Level: {a['risk_label']}")
        print(f"🔍 Top Symptom: {report['symptoms']['top_symptom_display']}")
        print(f"📊 Severity Score: {report['severity']['normalized']}/10")

    print(f"\n⚠️  {report['disclaimer']}")

    # حفظ الـ JSON
    output_path = args.output or 'screening_report.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Report saved: {output_path}")
    print(f"{'='*60}")

    return report


if __name__ == '__main__':
    main()
