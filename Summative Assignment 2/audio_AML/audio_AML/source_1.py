import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium", auto_download=["ipynb"])


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
    import sklearn
    import marimo as mo
    import shutil

    return os, pandas, pathlib, pb, shutil, sklearn


@app.cell
def _(os):
    # Get a list of the files on my drive
    file_list = os.listdir("C:/Users/Liam Holland/Documents/GitHub/audio_machine_learning/Summative Assignment 2/audio_AML/audio_AML/Medley-solos-DB")
    return (file_list,)


@app.cell
def _(file_list, pandas):
    # Convert into a dataframe so I can add a file_path column
    file_path_df = pandas.DataFrame(data=file_list)
    file_path_df = file_path_df.rename(columns={0:"file_path"})
    file_path_df["file_path"] = "./Medley-solos-DB/" + file_path_df["file_path"]

    # Extract the unique identifer so I can join on it
    file_path_df["uuid4"] = file_path_df["file_path"].str.extract(r"_.*_(.*).wav")
    return (file_path_df,)


@app.cell
def _(file_path_df, pandas):
    # Read in the metadata and join it to the file path
    metadata_df = pandas.read_csv(filepath_or_buffer=f"C:/Users/Liam Holland/Documents/GitHub/audio_machine_learning/Summative Assignment 2/audio_AML/audio_AML/Medley-solos-DB_metadata.csv")

    # Remove unneccary columns
    metadata_df = metadata_df.drop(["subset", "instrument_id","song_id"], axis=1, errors='ignore')

    # Remove electric guitar
    metadata_df = metadata_df[metadata_df["instrument"] != "distorted electric guitar"]

    metadata_df['uuid4'] = metadata_df['uuid4'].astype(str)
    file_path_df['uuid4'] = file_path_df['uuid4'].astype(str)

    joined_df = pandas.merge(left=metadata_df, right=file_path_df , how="left", on="uuid4")
    return (joined_df,)


@app.cell
def _(joined_df, pandas, sklearn):
    # Split the model into train test and val

    train, temp = sklearn.model_selection.train_test_split(joined_df, train_size = 0.8 , test_size = 0.2, shuffle=True, stratify = joined_df["instrument"] )

    test, val = sklearn.model_selection.train_test_split(temp, train_size = 0.5 , test_size = 0.5, shuffle=True, stratify = temp["instrument"] )

    # Limit the result to exactly 10 of each instrument
    train_filtered = train.groupby("instrument").head(10)
    train_filtered["test_train_val"] = "train"

    #Get 100 samples for Brittany
    brit_filtered = train.groupby("instrument").tail(15).reset_index(drop=True)

    # Limit the result to exactly 20 of each instrument
    test_filtered = test.groupby("instrument").head(2)
    test_filtered["test_train_val"] = "test"
    val_filtered = val.groupby("instrument").head(2)
    val_filtered["test_train_val"] = "val"

    filtered_df = pandas.concat([train_filtered, test_filtered, val_filtered]).reset_index(drop=True)
    return brit_filtered, filtered_df


@app.cell
def _(pathlib, shutil):
    def copy_files_from_df(df, path_column, destination_folder):
    
        # 1. Ensure the destination exists
        dest_path = pathlib.Path(destination_folder)
        dest_path.mkdir(parents=True, exist_ok=True)
    
        # 2. Iterate and copy
        for original_path_str in df[path_column]:
            source = pathlib.Path(original_path_str)
        
            if source.exists():
                # This copies to destination/filename.wav
                shutil.copy2(source, dest_path / source.name)
            else:
                print(f"Skipping: File not found at {source}")
            

    return (copy_files_from_df,)


@app.cell
def _(brit_filtered, copy_files_from_df):
    copy_files_from_df(brit_filtered, "file_path", "./105_raw_samples/")
    return


@app.cell
def _(pathlib, pb):
    def applyEffect( _effect, _effect_name , _wet_dry, _parameter, _parameter_value , _df_row , _folderOutput):

        # If it's a Reverb, we force dry_level to 0
        if hasattr(_effect, 'dry_level'):
            _effect.dry_level = 0.0
            _effect.wet_level = 1.0
        # If it's a Delay/Chorus, we force mix to 1.0 (all wet)
        if hasattr(_effect, 'mix'):
            _effect.mix = 1.0

        # To understand where these have come from help(pb.Reverb)

        board = pb.Pedalboard([_effect])

        filePath = f".\\{str(pathlib.Path(_df_row["file_path"]))}"

        #Output file path
        output_dir = pathlib.Path(_folderOutput)

        # Create the folder if it's missing
        output_dir.mkdir(parents=True, exist_ok=True)

        rows = []

        try:  
            for wet in  _wet_dry:

                for param in _parameter_value:

                    # Clear the reverb/delay buffers for the new file
                    board.reset()

                    # Change parameter dynamically
                    if hasattr(board[0], _parameter):
                        setattr(board[0], _parameter, param)
                    else:
                        print(f"Warning: {_effect_name} does not have a parameter named '{_parameter}'")

                    # Use / operator to join paths (handles slashes automatically)
                    pathOutput = str(output_dir / f"{_effect_name}_wet{wet}_{_parameter}{param}_{_df_row["uuid4"]}.wav")

                    # Open an audio file for reading, just like a regular file:
                    with pb.io.AudioFile(f'{filePath}') as f:

                      # Open an audio file to write to:
                      with pb.io.AudioFile(pathOutput, 'w', f.samplerate, f.num_channels) as o:

                        # Read one second of audio at a time, until the file is empty:
                        while f.tell() < f.frames:
                          chunk = f.read(f.samplerate)

                          # Run the audio through our pedalboard:
                          effected = board(chunk, f.samplerate, reset=False)

                          # # Mix
                          mix = wet * effected + (1 - wet) * chunk

                          # Write the output to our output file:
                          o.write(mix)

                    new_row  = _df_row.copy()
                    new_row["new_file_path"] = pathOutput
                    new_row ["effect_applied"] = f"{_effect_name}"
                    new_row ["wet_dry"] = f"{wet}"
                    new_row ["parameter"] = f"{_parameter}"
                    new_row ["parameter_value"] = f"{param}"
                    rows.append(new_row)

        except Exception as e:
            print(f"Error processing {filePath} with {_effect_name}: {e}")
            return [] # Return empty list so .extend() adds nothing

        return rows


    return (applyEffect,)


@app.cell
def _(applyEffect, filtered_df, pandas, pb):
    reverb = pb.Reverb(room_size = 0.2, 
                       damping = 0.5, 
                       width = 1.0, 
                       freeze_mode = 0.0)


    delay = pb.Delay( delay_seconds = 0.25, 
                      feedback = 0.3)

    distortion = pb.Distortion(drive_db = 14.0)

    chorus = pb.Chorus(rate_hz = 1.0, 
                       depth = 0.25, 
                       centre_delay_ms = 7.0, 
                       feedback = 0.2)

    rows_list = []

    for idx, sample in filtered_df.iterrows():
        rows_list.extend(applyEffect(reverb, "reverb", [0.0, 0.25, 0.5, 0.75], "room_size", [0.2, 0.8]  , sample, "reverb/"))
        rows_list.extend(applyEffect(delay, "delay",[0.0, 0.25, 0.5, 0.75], "delay_seconds", [0.25, 0.5] , sample, "delay/"))
        rows_list.extend(applyEffect(distortion, "distortion",[0.0, 0.25, 0.5, 0.75],"drive_db", [14 , 28], sample, "distortion/"))
        rows_list.extend(applyEffect(chorus, "chorus",[0.0, 0.25, 0.5, 0.75],"feedback", [0.2 , 0.6] , sample, "chorus/"))

    new_df = pandas.DataFrame(rows_list).reset_index(drop=True)
    pandas.DataFrame.to_csv(new_df, "train_test_val_mapping.csv")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
