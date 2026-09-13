import gradio as gr
import subprocess
import sys
import os
from pathlib import Path
import threading

def install_dependencies(device_choice):
    """Install the proper torch version and dependencies based on device choice."""
    if device_choice == "CUDA (NVIDIA GPU)":
        torch_index = "https://download.pytorch.org/whl/cu121"
    else:
        torch_index = "https://download.pytorch.org/whl/cpu"

    log_lines = []
    log_lines.append(f"Installing PyTorch for {device_choice}...")

    try:
        # Install torch with appropriate index
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "torch", f"--index-url={torch_index}"],
            capture_output=True,
            text=True,
            check=True
        )
        log_lines.append("✓ PyTorch installed successfully")

        # Install other dependencies
        log_lines.append("Installing openai-whisper, argostranslate, srt...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "openai-whisper", "argostranslate", "srt"],
            capture_output=True,
            text=True,
            check=True
        )
        log_lines.append("✓ All dependencies installed successfully")
        log_lines.append("\nReady to process videos!")
    except subprocess.CalledProcessError as e:
        log_lines.append(f"✗ Installation failed: {e.stderr}")
    except Exception as e:
        log_lines.append(f"✗ Error: {str(e)}")

    return "\n".join(log_lines)

PROJECT_DIR = Path(__file__).resolve().parent

def cleanup_leftovers(working_path_obj, local_video_path, copied, log_lines):
    """Clean up intermediate files (srt subtitles and copied input video)."""
    cleanup_dirs = {working_path_obj.parent, PROJECT_DIR}
    cleaned = set()
    for directory in cleanup_dirs:
        for srt_file in directory.glob(f"{working_path_obj.stem}*.srt"):
            if srt_file not in cleaned:
                try:
                    srt_file.unlink(missing_ok=True)
                    cleaned.add(srt_file)
                    log_lines.append(f"✓ Cleaned up intermediate subtitle: {srt_file.name}")
                except Exception as e:
                    log_lines.append(f"Could not remove {srt_file.name}: {e}")

    # Remove copied input video if it was copied
    if copied and local_video_path.exists():
        try:
            local_video_path.unlink(missing_ok=True)
            log_lines.append(f"✓ Cleaned up temporary copied video: {local_video_path.name}")
        except Exception as e:
            log_lines.append(f"Could not remove {local_video_path.name}: {e}")

def process_video(video_path, device_choice, to_lang, hardcode):
    """Process the video with subtitle generation."""
    if video_path is None:
        return None, gr.DownloadButton(value=None, visible=False), "Please upload a video file first."

    # Copy video to project directory to ensure it's accessible
    import shutil
    video_path_obj = Path(video_path)
    local_video_path = PROJECT_DIR / video_path_obj.name

    log_lines = []
    log_lines.append("Copying video to project directory...")

    copied = False
    try:
        shutil.copy2(video_path, str(local_video_path))
        log_lines.append(f"✓ Video copied to: {local_video_path}")
        working_video_path = str(local_video_path)
        copied = True
    except Exception as e:
        log_lines.append(f"✗ Failed to copy video: {str(e)}")
        log_lines.append("Attempting to use original path...")
        working_video_path = video_path

    working_path_obj = Path(working_video_path)

    # Map device choice to actual device string
    device_map = {"CUDA (NVIDIA GPU)": "cuda", "CPU": "cpu"}
    device = device_map.get(device_choice, "cpu")

    log_lines.append(f"Processing video: {working_video_path}")
    log_lines.append(f"Device: {device}")
    log_lines.append(f"Target language: {to_lang}")
    log_lines.append("Audio language will be auto-detected by Whisper...")

    try:
        # Build command - no from-lang since we're auto-detecting
        cmd = [
            sys.executable, str(PROJECT_DIR / "subtitle_pipeline.py"),
            working_video_path,
            "--to-lang", to_lang,
            "--device", device
        ]

        if hardcode:
            cmd.append("--hardcode")

        log_lines.append(f"Running: {' '.join(cmd)}")
        log_lines.append("")

        # Ensure environment has venv bin in PATH
        env = os.environ.copy()
        venv_bin = str(Path(sys.executable).parent)
        env["PATH"] = f"{venv_bin}:{env.get('PATH', '')}"

        # Run the pipeline
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(PROJECT_DIR),
            env=env
        )

        log_lines.append(result.stdout)
        if result.stderr:
            log_lines.append(result.stderr)

        if result.returncode == 0:
            # Find output file in working directory
            output_name = f"{working_path_obj.stem}_subbed{working_path_obj.suffix}"
            output_path = working_path_obj.parent / output_name

            found_output = None
            if output_path.exists():
                found_output = output_path
            else:
                matches = list(PROJECT_DIR.glob(f"{working_path_obj.stem}_subbed*"))
                if matches:
                    found_output = matches[0]

            # Cleanup intermediate leftovers (srt files, copied input video)
            cleanup_leftovers(working_path_obj, local_video_path, copied, log_lines)

            if found_output and found_output.exists():
                log_lines.append(f"\n✓ Done! Output saved to: {found_output}")
                return str(found_output), gr.DownloadButton(value=str(found_output), visible=True), "\n".join(log_lines)
            else:
                log_lines.append("\n✗ Processing succeeded but output file was not found.")
                return None, gr.DownloadButton(value=None, visible=False), "\n".join(log_lines)

        log_lines.append(f"\n✗ Processing failed with return code {result.returncode}")
        cleanup_leftovers(working_path_obj, local_video_path, copied, log_lines)
        return None, gr.DownloadButton(value=None, visible=False), "\n".join(log_lines)

    except Exception as e:
        log_lines.append(f"\n✗ Error: {str(e)}")
        cleanup_leftovers(working_path_obj, local_video_path, copied, log_lines)
        return None, gr.DownloadButton(value=None, visible=False), "\n".join(log_lines)

# Create Gradio interface
with gr.Blocks(title="Video Subtitle Generator") as demo:
    gr.Markdown("# 🎬 Video Subtitle Generator")
    gr.Markdown("Upload a video, choose your device, and get subtitled output!")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Step 1: Choose Device & Install Dependencies")
            device_choice = gr.Radio(
                choices=["CUDA (NVIDIA GPU)", "CPU"],
                value="CPU",
                label="Select Compute Device"
            )
            install_btn = gr.Button("📦 Install Dependencies", variant="primary")
            install_log = gr.Textbox(
                label="Installation Log",
                lines=8,
                max_lines=20,
                interactive=False
            )

            gr.Markdown("### Step 2: Configure Settings")
            gr.Markdown("**Audio Language:** 🎯 Auto-detected from video using Whisper")
            to_lang = gr.Dropdown(
                choices=[
                    ("Greek", "el"),
                    ("English", "en"),
                    ("Spanish", "es"),
                    ("French", "fr"),
                    ("German", "de"),
                    ("Italian", "it"),
                    ("Portuguese", "pt"),
                    ("Russian", "ru"),
                    ("Japanese", "ja"),
                    ("Chinese", "zh"),
                    ("Korean", "ko"),
                    ("Arabic", "ar"),
                    ("Hindi", "hi"),
                ],
                value="el",
                label="Target Language"
            )
            hardcode = gr.Checkbox(
                label="Hardcode subtitles into video (instead of soft subs)",
                value=False
            )

        with gr.Column(scale=1):
            gr.Markdown("### Step 3: Upload Video")
            video_input = gr.Video(
                label="Drop your video here",
                sources=["upload"]
            )
            process_btn = gr.Button("🎯 Generate Subtitles", variant="primary", size="lg")

            gr.Markdown("### Output")
            video_output = gr.Video(label="Subtitled Video")
            download_btn = gr.DownloadButton(
                label="📥 Download Subtitled Video",
                visible=False,
                variant="primary",
                size="lg"
            )
            process_log = gr.Textbox(
                label="Processing Log",
                lines=10,
                max_lines=25,
                interactive=False
            )

    # Wire up buttons
    install_btn.click(
        fn=install_dependencies,
        inputs=[device_choice],
        outputs=[install_log]
    )

    process_btn.click(
        fn=process_video,
        inputs=[video_input, device_choice, to_lang, hardcode],
        outputs=[video_output, download_btn, process_log]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
