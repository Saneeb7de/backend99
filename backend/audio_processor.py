from pathlib import Path
import ffmpeg  # Use the ffmpeg-python wrapper

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
    
    # --- FFmpeg Command Chain (Optimized for low-memory environments) ---
    # The 'loudnorm' filter is excluded as it is very resource-intensive.
    # 1. -i {input_path}: Specifies the input file.
    # 2. -af silenceremove...: Removes silences longer than 2 seconds.
    #    NOTE: The -40dB threshold might be too aggressive for quiet microphones,
    #    potentially removing all audio. Consider making this less strict if issues persist.
    # 3. -c:a libopus: Encodes the output audio with the Opus codec.
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
        # The full error from ffmpeg is printed for easier debugging
        print(e.stderr.decode())
        raise e

    # --- ROBUSTNESS CHECK ---
    # Before splitting, verify the processed file was created and is not empty.
    # This prevents the 'End of file' error if silenceremove stripped everything.
    try:
        if not processed_path.exists() or processed_path.stat().st_size == 0:
            print(f"⚠️ FFMPEG Processor Warning: Processed file '{processed_path.name}' is empty or missing.")
            print("Skipping splitting step. This usually means the original recording was silent or too short.")
            # Return a list containing only the (empty) processed file so it can be cleaned up.
            return [processed_path]
        else:
            print(f"✅ Processed audio is valid (size: {processed_path.stat().st_size} bytes). Proceeding to split.")
    except FileNotFoundError:
        # This is a fallback, but the .exists() check should prevent it.
        print(f"⚠️ FFMPEG Processor Warning: Processed file '{processed_path.name}' not found after processing.")
        return []

    # --- FFmpeg Splitting Command (Corrected and Robust) ---
    # This part is fast as it copies the stream without re-encoding.
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
             # The original processed file is the only thing to return in this case.
             return [processed_path]

        print(f"FFMPEG Processor: Split into {len(final_chunk_paths)} uploadable chunks.")
        
        # Add the intermediate processed file to the list for eventual cleanup.
        final_chunk_paths.append(processed_path)
        
        return final_chunk_paths

    except ffmpeg.Error as e:
        print("❌ FFMPEG Error during splitting:")
        print(e.stderr.decode())
        raise e