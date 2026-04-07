"""
Camera Calibration Pipeline - Run All Steps
This script provides an easy way to run the complete camera calibration pipeline.
"""

import os
import sys
import argparse
import json


def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════╗
║           Camera Calibration & Distortion Correction          ║
╚══════════════════════════════════════════════════════════════╝
    """)


def check_dependencies():
    """Check if required packages are installed."""
    required = ['cv2', 'numpy']
    missing = []
    
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print("필요한 패키지가 없습니다:")
        for pkg in missing:
            pkg_name = 'opencv-python' if pkg == 'cv2' else pkg
            print(f"  - {pkg} (설치: pip install {pkg_name})")
        return False
    
    return True


def step2_calibrate(video_path, board_size=(9, 6), square_size=25.0):
    """Step 2: Perform camera calibration."""
    print("\n" + "="*60)
    print("STEP 2: Camera Calibration")
    print("="*60)
    
    from camera_calibration import (
        extract_frames_from_video, 
        calibrate_camera, 
        save_calibration_result, 
        print_calibration_result,
        generate_readme_content,
        CHESSBOARD_SIZE, SQUARE_SIZE
    )
    
    # Update settings
    import camera_calibration
    camera_calibration.CHESSBOARD_SIZE = board_size
    camera_calibration.SQUARE_SIZE = square_size
    
    # Extract frames and find chessboard
    obj_points, img_points, img_size = extract_frames_from_video(video_path, frame_interval=30)
    
    if len(obj_points) == 0:
        print("Error: No valid chessboard images found!")
        return None
    
    # Calibrate
    result = calibrate_camera(obj_points, img_points, img_size)
    
    if result:
        print_calibration_result(result)
        save_calibration_result(result, "calibration_result.json")
        
        # Generate README content
        readme_content = generate_readme_content(result)
        print("\n" + "="*60)
        print("README.md Content (Camera Calibration section):")
        print("="*60)
        print(readme_content)
        
        return result
    
    return None


def step3_correct_distortion(calibration_path, input_path):
    """Step 3: Apply distortion correction."""
    print("\n" + "="*60)
    print("STEP 3: Distortion Correction")
    print("="*60)
    
    from distortion_correction import (
        load_calibration, 
        process_video, 
        process_image
    )
    
    camera_matrix, dist_coeffs, calib_result = load_calibration(calibration_path)
    
    os.makedirs("output", exist_ok=True)
    
    ext = os.path.splitext(input_path)[1].lower()
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    
    if ext in video_extensions:
        result_path = process_video(input_path, "output", camera_matrix, dist_coeffs)
    else:
        result_path = process_image(input_path, "output", camera_matrix, dist_coeffs)
    
    print(f"\nDistortion correction complete!")
    print(f"Output saved in: output/")
    
    return result_path


def generate_full_readme(calibration_result, comparison_image_path=None):
    """Generate complete README.md file."""
    
    readme = """# My Camera Calibrator

A Python-based camera calibration tool using OpenCV. This project performs camera calibration using chessboard patterns and applies lens distortion correction.

## Features

- Camera calibration using chessboard pattern detection
- Lens distortion correction for images and videos
- Live preview mode for real-time distortion correction
- Video recorder with alignment grid for calibration footage

## Requirements

- Python 3.7+
- OpenCV (`pip install opencv-python`)
- NumPy (`pip install numpy`)

## Usage

### 1. Record Calibration Video

```bash
python video_recorder.py -o chessboard.avi
```

Controls:
- SPACE: Start/Stop recording
- G: Toggle grid overlay
- Q: Quit

### 2. Run Camera Calibration

```bash
python camera_calibration.py chessboard.avi --board-size 9,6 --square-size 25.0
```

### 3. Apply Distortion Correction

```bash
# For images
python distortion_correction.py calibration_result.json -i image.jpg -o output/

# For videos
python distortion_correction.py calibration_result.json -i video.mp4 -o output/

# Live preview
python distortion_correction.py calibration_result.json --live
```

### Quick Run (All Steps)

```bash
python run_all.py
```

"""
    
    if calibration_result:
        readme += f"""
## Camera Calibration Results

### Intrinsic Parameters (Camera Matrix)
| Parameter | Value |
|-----------|-------|
| fx (focal length x) | {calibration_result['fx']:.4f} pixels |
| fy (focal length y) | {calibration_result['fy']:.4f} pixels |
| cx (principal point x) | {calibration_result['cx']:.4f} pixels |
| cy (principal point y) | {calibration_result['cy']:.4f} pixels |

### Distortion Coefficients
| Parameter | Value |
|-----------|-------|
| k1 (radial) | {calibration_result['k1']:.6f} |
| k2 (radial) | {calibration_result['k2']:.6f} |
| p1 (tangential) | {calibration_result['p1']:.6f} |
| p2 (tangential) | {calibration_result['p2']:.6f} |
| k3 (radial) | {calibration_result['k3']:.6f} |

### Calibration Quality
- **RMSE (Reprojection Error)**: {calibration_result['rmse']:.6f} pixels
- **Number of images used**: {calibration_result['num_images_used']}
- **Image size**: {calibration_result['image_size'][0]}x{calibration_result['image_size'][1]}

"""
    
    readme += """
## Distortion Correction Demo

### Before and After Comparison
"""
    
    if comparison_image_path:
        # Use relative path
        rel_path = os.path.basename(comparison_image_path)
        readme += f"""
![Distortion Correction Comparison](output/{rel_path})

*Left: Original (distorted) | Right: Corrected (undistorted)*
"""
    else:
        readme += """
*[Add your comparison image/video here]*

Example:
```
![Comparison](output/comparison.jpg)
```
"""
    
    readme += """
## Chessboard Pattern

For calibration, use a standard chessboard pattern. You can:
- Print the pattern from: https://markhedleyjones.com/projects/calibration-checkerboard-collection
- Use a 9x6 inner corners pattern (default)
- Attach the printed pattern to a flat, rigid surface

## References

- [OpenCV Camera Calibration Tutorial](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)
- [Zhang's Camera Calibration Method](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr98-71.pdf)

## License

MIT License
"""
    
    return readme


def main():
    print_banner()
    
    if not check_dependencies():
        sys.exit(1)
    
    parser = argparse.ArgumentParser(description='Camera Calibration Pipeline')
    parser.add_argument('--video', '-v', help='Path to calibration video')
    parser.add_argument('--board-size', default='9,6', help='Chessboard size (cols,rows)')
    parser.add_argument('--square-size', type=float, default=25.0, help='Square size in mm')
    parser.add_argument('--skip-record', action='store_true', help='Skip video recording step')
    parser.add_argument('--calibration', '-c', help='Path to existing calibration JSON')
    parser.add_argument('--input', '-i', help='Input for distortion correction')
    
    args = parser.parse_args()
    
    # Parse board size
    board_size = tuple(map(int, args.board_size.split(',')))
    
    calibration_result = None
    calibration_path = args.calibration or "calibration_result.json"
    comparison_path = None
    
    # Step 2: Calibration
    if args.video and not args.calibration:
        calibration_result = step2_calibrate(args.video, board_size, args.square_size)
    elif args.calibration and os.path.exists(args.calibration):
        with open(args.calibration, 'r') as f:
            calibration_result = json.load(f)
        print(f"\n기존 캘리브레이션 로드: {args.calibration}")
    
    # Step 3: Distortion correction
    if os.path.exists(calibration_path):
        input_file = args.input or args.video
        if input_file:
            comparison_path = step3_correct_distortion(calibration_path, input_file)
        else:
            print("\n왜곡 보정을 위한 입력 파일이 없습니다.")
            print("사용법: python distortion_correction.py calibration_result.json -i <이미지_또는_동영상>")
    
    # Generate README
    if calibration_result:
        readme_content = generate_full_readme(calibration_result, comparison_path)
        with open("README.md", 'w') as f:
            f.write(readme_content)
        print("\n" + "="*60)
        print("README.md 생성 완료!")
        print("="*60)
    
    print("\n" + "="*60)
    print("완료!")
    print("="*60)
    print("""
다음 단계:
1. 체스보드 영상으로 캘리브레이션 실행
2. 이미지/동영상에 왜곡 보정 적용
3. README.md에 결과 및 스크린샷 업데이트
4. GitHub에 푸시!
    """)


if __name__ == "__main__":
    main()
