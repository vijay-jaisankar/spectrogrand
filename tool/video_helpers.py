"""
    Collection of helper functions pertaining to the video domain of spectrogrand
"""
from moviepy.editor import AudioFileClip, ImageClip, ImageSequenceClip
from typing import Optional

"""
    @method generate_and_save_video_dynamic
        Generate and save a dynamic video for a given audio file and image folder 
    @param audio_file_path: Path to the background audio file
    @param image_dir_path: Path to the directory containing the list of images to be included in the video (@note Expected file format: *_{idx}.* - the files will be sorted based on `idx`.)
    @param output_video_path: Path to the file where the video is to be stored
"""
def generate_and_save_video_dynamic(audio_file_path:str, image_dir_path:str, output_video_path:str) -> Optional[str]:
    try:
        # Create audio clip
        with AudioFileClip(filename=audio_file_path) as audio_clip:
            audio_duration = audio_clip.duration

            # Create image sequence clip
            with ImageSequenceClip(sequence=image_dir_path, fps = 10) as image_sequence_clip:
                image_sequence_clip = image_sequence_clip.set_duration(audio_duration)
                image_sequence_clip = image_sequence_clip.set_audio(audio_clip)
                image_sequence_clip = image_sequence_clip.set_fps(10)

                # Export the clip
                image_sequence_clip.write_videofile(output_video_path)
                return output_video_path
    except Exception as e:
        print(f"Error while generating dynamic video clip: {e}")
        return None

"""
    @method generate_and_save_video_static
        Generate and save a static video for a given audio file and image file 
    @param audio_file_path: Path to the background audio file
    @param image_file_path: Path to the image file
    @param output_video_path: Path to the file where the video is to be stored
"""
def generate_and_save_video_static(audio_file_path:str, image_file_path:str, output_video_path:str) -> Optional[str]:
    try:
        # Create audio clip
        with AudioFileClip(filename=audio_file_path) as audio_clip:
            audio_duration = audio_clip.duration

            # Create image clip
            with ImageClip(img=image_file_path) as image_clip:
                image_clip = image_clip.set_duration(audio_duration) # @note ref: https://stackoverflow.com/questions/75414756/combine-image-and-audio-together-using-moviepy-in-python
                image_clip = image_clip.set_audio(audio_clip)
                image_clip = image_clip.set_fps(10)

                # Export the clip
                image_clip.write_videofile(output_video_path)
                return output_video_path
    except Exception as e:
        print(f"Error while generating static video clip: {e}")
        return None