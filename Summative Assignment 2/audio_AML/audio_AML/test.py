import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import glob
    import librosa
    import numpy
    import sys
    import pandas
    import re
    import pedalboard as pb
    import pathlib

    return os, pandas, pathlib, pb


@app.cell
def _(os):
    file_list = os.listdir("C:/Users/Liam Holland/Documents/GitHub/audio_machine_learning/Summative Assignment 2/audio_AML/audio_AML/Medley-solos-DB")
    return (file_list,)


@app.cell
def _(file_list, pandas):
    file_path_df = pandas.DataFrame(data=file_list)
    file_path_df = file_path_df.rename(columns={0:"file_path"})
    file_path_df["file_path"] = "./Medley-solos-DB/" + file_path_df["file_path"]
    return (file_path_df,)


@app.cell
def _(file_path_df):
    file_path_df["uuid4"] = file_path_df["file_path"].str.extract(r"_.*_(.*).wav")
    return


@app.cell
def _(pandas):
    metadata_df = pandas.read_csv(filepath_or_buffer=f"C:/Users/Liam Holland/Documents/GitHub/audio_machine_learning/Summative Assignment 2/audio_AML/audio_AML/Medley-solos-DB_metadata.csv")
    return (metadata_df,)


@app.cell
def _(file_path_df, metadata_df):
    metadata_df['uuid4'] = metadata_df['uuid4'].astype(str)
    file_path_df['uuid4'] = file_path_df['uuid4'].astype(str)
    return


@app.cell
def _(file_path_df, metadata_df, pandas):
    joined_df = pandas.merge(left=file_path_df, right=metadata_df , how="left", on="uuid4")
    return (joined_df,)


@app.cell
def _(joined_df):
    joined_df.head(n=5)
    return


@app.cell
def _(pathlib, pb):
    def applyReverb(_filePath, _uuid4 , _folderOutput):

        board = pb.Pedalboard( [pb.Reverb(room_size = 0.25)] )

        filePath = f".\\{str(pathlib.Path(_filePath))}"
        folderOutput = f".\\{str(pathlib.Path(_folderOutput))}" 

        pathOutput = f"{folderOutput}./REVERB_{_uuid4}.wav"

        # Open an audio file for reading, just like a regular file:
        with pb.io.AudioFile(f'{filePath}') as f:
      
          # Open an audio file to write to:
          with pb.io.AudioFile(pathOutput, 'w', f.samplerate, f.num_channels) as o:
      
            # Read one second of audio at a time, until the file is empty:
            while f.tell() < f.frames:
              chunk = f.read(f.samplerate)
          
              # Run the audio through our pedalboard:
              effected = board(chunk, f.samplerate, reset=False)
          
              # Write the output to our output file:
              o.write(effected)

    return (applyReverb,)


@app.cell
def _(applyReverb):
    applyReverb(f"./Medley-solos-DB/Medley-solos-DB_test-0_003d41a8-afad-501f-fbc3-d406dc694ceb.wav","003d41a8-afad-501f-fbc3-d406dc694ceb","./reverb/")
    return


@app.cell
def _(applyReverb, joined_df):
    for i in range(20):

        # Use iloc to get integer based position
        _path = str(joined_df["file_path"].iloc[i])
        _id = str(joined_df["uuid4"].iloc[i])
    
        applyReverb(_path, _id, "reverb/")
    return


@app.cell
def _(pb):
    help(pb.Reverb)
    return


if __name__ == "__main__":
    app.run()
