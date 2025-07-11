# # backend/audio_processor.py

# from pathlib import Path
# import ffmpeg # Use the ffmpeg-python wrapper

# def process_and_split_audio(input_path: Path, session_id: str, temp_dir: Path) -> list[Path]:
#     """
#     Processes a long audio file and splits it into chunks using FFmpeg.
#     This entire process is memory-efficient as it operates on files on disk.

#     - Normalizes audio volume using the 'loudnorm' filter.
#     - Removes long silences using the 'silenceremove' filter.
#     - Splits the processed audio into ~55-second OGG chunks.
#     - Returns a list of paths to the final chunks.
#     """
#     print(f"FFMPEG Processor: Starting processing for {input_path.name}")
    
#     processed_path = temp_dir / f"{session_id}_processed.ogg"
    
#     # --- FFmpeg Command Chain ---
#     # This looks complex, but it's a highly efficient single command.
#     # 1. -i {input_path}: Specifies the input file.
#     # 2. af loudnorm: Normalizes the audio level.
#     # 3. af silenceremove...: Removes silences longer than 2 seconds.
#     # 4. -c:a libopus: Encodes the output audio with the Opus codec.
#     try:
#         (
#             ffmpeg
#             .input(str(input_path))
#             .filter('loudnorm')
#             .filter('silenceremove', start_periods=1, start_duration=0, start_threshold='-40dB', stop_periods=-1, stop_duration=2, stop_threshold='-40dB')
#             .output(str(processed_path), acodec='libopus')
#             .overwrite_output()
#             .run(capture_stdout=True, capture_stderr=True)
#         )
#         print(f"FFMPEG Processor: Audio normalization and silence removal complete for {processed_path.name}")
#     except ffmpeg.Error as e:
#         print("❌ FFMPEG Error during processing:")
#         print(e.stderr.decode())
#         raise e

#     # --- FFmpeg Splitting Command ---
#     # Now, split the processed file into segments without re-encoding.
#     # 1. -i {processed_path}: Input is the processed file.
#     # 2. -f segment: Use the segment muxer.
#     # 3. -segment_time 55: Create segments of 55 seconds.
#     # 4. -c copy: Copy the audio stream without re-encoding (very fast).
#     chunk_filename_pattern = temp_dir / f"{session_id}_upload_chunk_%03d.ogg"
#     try:
#         (
#             ffmpeg
#             .input(str(processed_path))
#             .output(str(chunk_filename_pattern), f='segment', segment_time=55, c='copy')
#             .overwrite_output()
#             .run(capture_stdout=True, capture_stderr=True)
#         )
#         # Find the chunks that were actually created
#         final_chunk_paths = sorted(temp_dir.glob(f"{session_id}_upload_chunk_*.ogg"))
#         print(f"FFMPEG Processor: Split into {len(final_chunk_paths)} uploadable chunks.")
        
#         # Add the intermediate processed file to the cleanup list
#         final_chunk_paths.append(processed_path)
        
#         return final_chunk_paths

#     except ffmpeg.Error as e:
#         print("❌ FFMPEG Error during splitting:")
#         print(e.stderr.decode())
#         raise e

# backend/audio_processor.py

from pathlib import Path
import ffmpeg # Use the ffmpeg-python wrapper

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
        print("❌ FFMPEG Error during processing:")
        # The full error from ffmpeg is printed for easier debugging
        print(e.stderr.decode())
        raise e

    # --- FFmpeg Splitting Command (Corrected and Robust) ---
    # This part is fast as it copies the stream without re-encoding.
    # 1. -i {processed_path}: Input is the processed file.
    # 2. -f segment: Use the segment muxer.
    # 3. -segment_time 55: Create segments of approximately 55 seconds.
    # 4. -c copy: Copy the audio stream without re-encoding.
    # 5. -reset_timestamps 1: Ensures each chunk starts with a fresh timestamp.
    
    # Bug Fix: Corrected the filename pattern to include an underscore.
    chunk_filename_pattern = temp_dir / f"{session_id}_upload_chunk_%03d.ogg"
    
    try:
        (
            ffmpeg
            .input(str(processed_path))
            .output(str(chunk_filename_pattern), f='segment', segment_time=55, c='copy', reset_timestamps=1)
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )

        # Bug Fix: Corrected the glob pattern to match the output pattern.
        final_chunk_paths = sorted(temp_dir.glob(f"{session_id}_upload_chunk_*.ogg"))
        
        if not final_chunk_paths:
             print("⚠️ FFMPEG Processor Warning: No chunks were created. The audio might be too short.")
             # If the original processed file is small enough, maybe you want to use it directly.
             # This depends on your application's logic.
             # For now, we'll return an empty list if no chunks are made.
             return []

        print(f"FFMPEG Processor: Split into {len(final_chunk_paths)} uploadable chunks.")
        
        # Add the intermediate processed file to the list for eventual cleanup.
        final_chunk_paths.append(processed_path)
        
        return final_chunk_paths

    except ffmpeg.Error as e:
        print("❌ FFMPEG Error during splitting:")
        print(e.stderr.decode())
        raise e