from pathlib import Path
import ffmpeg  # Use the ffmpeg-python wrapper

MINIMUM_FILE_SIZE_BYTES = 512 

def process_and_split_audio(input_path: Path, session_id: str, temp_dir: Path) -> list[Path]:
    """
    Processes a long audio file and splits it into chunks using FFmpeg.
    This entire process is memory-efficient as it operates on files on disk.

    - Removes long silences using the 'silenceremove' filter.
    - Splits the processed audio into ~55-second OGG chunks encoded with Opus.
    - Returns a list of paths to the final chunks and the intermediate file for cleanup.
    """
    print(f"FFMPEG Processor: Starting processing for {input_path.name}")
    
    processed_path = temp_dir / f"{session_id}_processed.ogg"
    
    try:
        (
            ffmpeg
            .input(str(input_path))
            .filter('silenceremove', start_periods=1, start_duration=0, start_threshold='-40dB', stop_periods=-1, stop_duration=2, stop_threshold='-40dB')
            .output(str(processed_path), acodec='libopus')
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        print(f"FFMPEG Processor: Audio silence removal complete for {processed_path.name}")
    except ffmpeg.Error as e:
        print("❌ FFMPEG Error during processing (silence removal):")
        print(e.stderr.decode())
        raise e

    # --- MORE ROBUST CHECK ---
    # Now we check if the file is large enough to contain actual audio data,
    # not just a header.
    try:
        file_size = processed_path.stat().st_size if processed_path.exists() else 0
        
        if file_size < MINIMUM_FILE_SIZE_BYTES:
            print(f"⚠️ FFMPEG Processor Warning: Processed file '{processed_path.name}' is too small ({file_size} bytes).")
            print("Skipping splitting step. This usually means the original recording was silent or too short.")
            return [processed_path] if processed_path.exists() else []
        else:
            print(f"✅ Processed audio is valid (size: {file_size} bytes). Proceeding to split.")
    except FileNotFoundError:
        print(f"⚠️ FFMPEG Processor Warning: Processed file '{processed_path.name}' not found after processing.")
        return []

    # --- FFmpeg Splitting Command ---
    chunk_filename_pattern = temp_dir / f"{session_id}_upload_chunk_%03d.ogg"
    
    try:
        (
            ffmpeg
            .input(str(processed_path))
            .output(str(chunk_filename_pattern), f='segment', segment_time=55, c='copy', reset_timestamps=1)
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )

        final_chunk_paths = sorted(temp_dir.glob(f"{session_id}_upload_chunk_*.ogg"))
        
        if not final_chunk_paths:
             print("⚠️ FFMPEG Processor Warning: No chunks were created. The audio might be too short for a full chunk.")
             return [processed_path]

        print(f"FFMPEG Processor: Split into {len(final_chunk_paths)} uploadable chunks.")
        
        final_chunk_paths.append(processed_path)
        
        return final_chunk_paths

    except ffmpeg.Error as e:
        print("❌ FFMPEG Error during splitting:")
        print(e.stderr.decode())
        raise e