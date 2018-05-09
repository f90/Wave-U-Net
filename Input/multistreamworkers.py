import numpy as np
import Input
from Sample import Sample
from soundfile import SoundFile
import librosa
import pickle as pkl
import os

class MultistreamWorker_GetSpectrogram:
    @staticmethod
    def run(communication_queue, exit_flag, options):
        '''
        Worker method that reads audio from a given file list and appends the processed spectrograms to the cache queue.
        :param communication_queue: Queue of the cache from which examples are added to the cache
        :param exit_flag: Flag to indicate when to exit the process
        :param options: Audio processing parameters and file list
        '''

        filename_list = options["file_list"]
        num_files = len(filename_list)

        # Alternative method: Load data in completely into RAM. Use only when data is small enough to fit into RAM and num_workers=1 in this case!
        #n_fft = options['num_fft']
        #hop_length = options['num_hop']

        # Re-seed RNG for this process
        np.random.seed()

        '''
        if os.path.exists("data.pkl"):
            with open("data.pkl", "r") as f:
                data = pkl.load(f)
        else:
            data = list()
            for item in filename_list:
                item_list = list()
                for sample in item[1:]:
                    print(sample.path)
                    audio, _ = librosa.load(sample.path, sr=options["expected_sr"], mono=options["mono_downmix"], res_type="kaiser_fast")
                    if len(audio.shape) == 1:
                        audio = np.expand_dims(audio, axis=0)
                    item_list.append(audio.T)
                data.append(np.array(item_list))
            with open("data.pkl", "wb") as f:
                pkl.dump(data, f)

        print("Finished loading data")
        
        duration_frames = int(options["duration"] * options["expected_sr"])

        while not exit_flag.is_set():
            try:# Decide which element to read next
                id_file_to_read = np.random.randint(num_files)
                sample = data[id_file_to_read]
                num_frames = len(sample[0])

                if duration_frames > num_frames:
                    print("Could not use sample " + str(id_file_to_read) + " since it is only " + str(float(num_frames) / float(options["expected_sr"])) + " s long, needed " + str(float(duration_frames) / float(options["expected_sr"])))

                start_source_pos = np.random.randint(num_frames-duration_frames)
                end_source_pos = start_source_pos + duration_frames
                start_mix_pos = max(start_source_pos - options["pad_frames"], 0)
                pad_mix_front = max(options["pad_frames"] - start_source_pos, 0)
                end_mix_pos = min(end_source_pos + options["pad_frames"], num_frames)
                pad_mix_back = max(end_source_pos + options["pad_frames"] - num_frames, 0)

                res = list()
                if options["augmentation"]:  # Random attenuation of source signals
                    mix_audio = np.zeros([duration_frames+2*options["pad_frames"], options["num_channels"]], np.float32)
                    for i in range(len(sample)):
                        amped_audio = Input.random_amplify(sample[i][start_mix_pos:end_mix_pos,:])
                        amped_audio = np.pad(amped_audio, [(pad_mix_front, pad_mix_back), (0,0)], "constant", constant_values=0.0)
                        mix_audio += amped_audio
                        if options["pad_frames"] > 0:
                            amped_audio = amped_audio[options["pad_frames"]:-options["pad_frames"]]
                        res.append(amped_audio)

                    res.insert(0, mix_audio)

                communication_queue.put(res)
            except Exception as e:
                print(e)
                print("Error while computing spectrogram. Skipping file.")


        '''
        while not exit_flag.is_set():
            # Decide which element to read next
            id_file_to_read = np.random.randint(num_files)
            item = filename_list[id_file_to_read]

            # Calculate the required amounts of padding
            duration_frames = int(options["duration"] * options["expected_sr"])
            padding_duration = options["padding_duration"]

            try:
                if isinstance(item, Sample):  # Single audio file: Use metadata to read section from it
                    metadata = [item.sample_rate, item.channels, item.duration]
                    audio, _, _, _ = Input.readAudio(item.path, offset=None, duration=options["duration"],
                                                     sample_rate=options["expected_sr"], pad_frames=options["pad_frames"],
                                                     metadata=metadata, mono=options["mono_downmix"])
                    #TF_rep, _ = Input.audioFileToSpectrogram(item.path, expected_sr=options["expected_sr"], offset=None,
                    #                                         duration=options["duration"], fftWindowSize=n_fft,
                    #                                         hopSize=hop_length,
                    #                                         padding_duration=options["padding_duration"],
                    #                                         metadata=metadata)
                    #TF_rep = np.ndarray.astype(TF_rep, np.float32)  # Cast to float32
                    #communication_queue.put(Input.random_amplify(TF_rep))
                    communication_queue.put(audio)

                elif isinstance(item,
                                float):  # This means the track is a (not as file existant) silence track so we insert a zero spectrogram
                    #TF_rep = np.zeros((n_fft / 2 + 1, duration_frames), dtype=np.float32)
                    #TF_rep = np.ndarray.astype(TF_rep, np.float32)  # Cast to float32
                    #communication_queue.put(Input.random_amplify(TF_rep))
                    communication_queue.put(np.zeros(duration_frames, np.float32))
                else:
                    assert (hasattr(item, '__iter__')) # Supervised case: Item is a list of files to read, starting with the mixture
                    # We want to get the spectrogram of the mixture (first entry in list), and of the sources and store them in cache as one training sample
                    sample = list()
                    #TODO under assumption that sources are additive we can only load sources in, then add them to form mixture!
                    file = item[0]
                    metadata = [file.sample_rate, file.channels, file.duration]
                    mix_audio, mix_sr, source_start_frame, source_end_frame, start_read, end_read = Input.readAudio(file.path, offset=None, duration=options[ "duration"],
                                                                                              sample_rate=options[ "expected_sr"],
                                                                                              pad_frames=options["pad_frames"],
                                                                                              metadata=metadata,
                                                                                              mono = options["mono_downmix"])
                    #mix_mag, _ = Input.audioFileToSpectrogram(mix_audio, fftWindowSize=n_fft, hopSize=hop_length)
                    #sample.append(mix_mag)
                    sample.append(mix_audio)

                    for file in item[1:]:
                        if isinstance(file, Sample):
                            #mag, _ = Input.audioFileToSpectrogram(file.path, expected_sr=options["expected_sr"], fftWindowSize=n_fft, hopSize=hop_length, buffer=True)
                            if options["augmentation"]:
                                source_audio, _ = Input.readWave(file.path, start_read, end_read,
                                                                 sample_rate=options["expected_sr"],
                                                                 mono=options["mono_downmix"])
                            else:
                                source_audio, _ = Input.readWave(file.path, source_start_frame, source_end_frame, sample_rate=options["expected_sr"], mono=options["mono_downmix"])
                            #source_audio = Input.random_amplify(source_audio)
                            #mag, _ = Input.audioFileToSpectrogram(source_audio, fftWindowSize=n_fft, hopSize=hop_length)
                            #mag = mag[:, :options["output_shape"][2]]
                        else:
                            assert (isinstance(file, float))  # This source is silent in this track
                            #padding_frames = (options["input_shape"][2] - options["output_shape"][2]) / 2  # Number of spectrogram frames to insert context in each direction
                            #source_shape = [mix_mag.shape[0], mix_mag.shape[1] - padding_frames * 2]
                            #mag = np.zeros(source_shape, dtype=np.float32)  # Therefore insert zeros
                            source_audio = np.zeros(mix_audio.shape, np.float32)
                        sample.append(source_audio)

                    # Check correct number of output channels
                    try:
                        assert (sample[0].shape[1] == options["num_channels"]
                                and sample[1].shape[1] == options["num_channels"]
                                and sample[2].shape[1] == options["num_channels"])
                    except Exception as e:
                        print("WARNING: Song " + file.path + " seems to be mono, will duplicate channels to convert into stereo for training!")
                        print("Channels for mix and sources" + str([sample[i].shape[1] for i in range(len(sample))]))

                    if options["augmentation"]: # Random attenuation of source signals
                        mix_audio = np.zeros(sample[0].shape, np.float32)
                        for i in range(1, len(sample)):
                            amped_audio = Input.random_amplify(sample[i])
                            mix_audio += amped_audio
                            if options["pad_frames"] > 0:
                                amped_audio = amped_audio[options["pad_frames"]:-options["pad_frames"]]
                            sample[i] = amped_audio

                        sample[0] = mix_audio

                    communication_queue.put(sample)
            except Exception as e:
                print(e)
                print("Error while computing spectrogram. Skipping file.")


        # This is necessary so that this process does not block. In particular, if there are elements still in the queue
        # from this process that were not yet 'picked up' somewhere else, join and terminate called on this process will
        # block
        communication_queue.cancel_join_thread()
