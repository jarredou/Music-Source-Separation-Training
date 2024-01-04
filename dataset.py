# coding: utf-8
__author__ = 'Roman Solovyev (ZFTurbo): https://github.com/ZFTurbo/'


import os
import random
import numpy as np
import torch
import soundfile as sf
import pickle
import time
from tqdm import tqdm
from glob import glob
import audiomentations as AU
import pedalboard as PB
#import torch_audiomentations as TAU

def load_chunk(path, length, chunk_size, offset=None):
    if chunk_size <= length:
        if offset is None:
            offset = np.random.randint(length - chunk_size + 1)
        x = sf.read(path, dtype='float32', start=offset, frames=chunk_size)[0]
    else:
        x = sf.read(path, dtype='float32')[0]
        pad = np.zeros([chunk_size - length, 2])
        x = np.concatenate([x, pad])
    return x.T


def get_transforms_simple(instr):
    if instr == 'vocals':
        augment = AU.Compose([
            AU.PitchShift(min_semitones=-5, max_semitones=5, p=0.1),
            AU.PolarityInversion(p=0.5),
            AU.SevenBandParametricEQ(min_gain_db = -9, max_gain_db=9, p=0.25),
            AU.TanhDistortion(min_distortion=0.1, max_distortion=0.7, p=0.1),
        ], p=1.0)
    elif instr == 'bass':
        augment = AU.Compose([
            AU.PitchShift(min_semitones=-2, max_semitones=2, p=0.1),
            AU.PolarityInversion(p=0.5),
            AU.SevenBandParametricEQ(min_gain_db = -3, max_gain_db=6, p=0.25),
            AU.TanhDistortion(min_distortion=0.1, max_distortion=0.5, p=0.2),
        ], p=1.0)
    elif instr == 'drums':
        augment = AU.Compose([
            AU.PitchShift(min_semitones=-5, max_semitones=5, p=0.33),
            AU.PolarityInversion(p=0.5),
            AU.SevenBandParametricEQ(min_gain_db = -9, max_gain_db=9, p=0.25),
            AU.TanhDistortion(min_distortion=0.1, max_distortion=0.6, p=0.33),
        ], p=1.0)
    else:
        augment = AU.Compose([
            AU.PitchShift(min_semitones=-3, max_semitones=3, p=0.1),
            AU.PolarityInversion(p=0.5),
            AU.TanhDistortion(min_distortion=0.1, max_distortion=0.4, p=0.25),
            AU.SevenBandParametricEQ(min_gain_db = -9, max_gain_db=9, p=0.25),
        ], p=1.0)
    return augment


class MSSDataset(torch.utils.data.Dataset):
    def __init__(self, config, data_path, metadata_path="metadata.pkl", dataset_type=1):
        self.config = config
        self.dataset_type = dataset_type # 1, 2, 3 or 4
        self.instruments = instruments = config.training.instruments

        # Augmentation block
        self.aug = False
        if config.training.augmentation == 1:
            print('Use augmentation for training')
            self.aug = True

        self.mp3_aug = None
        if config.training.use_mp3_compress:
            self.mp3_aug = AU.Compose([
                AU.Mp3Compression(min_bitrate=32, max_bitrate=320, backend="lameenc", p=0.1),
            ], p=1.0)

        self.augment_func = dict()
        for instr in self.instruments:
            self.augment_func[instr] = None
        if self.config.training.augmentation_type == "simple1":
            print('Use simple1 set of augmentations')
            for instr in self.instruments:
                self.augment_func[instr] = get_transforms_simple(instr)

        # metadata_path = data_path + '/metadata'
        try:
            metadata = pickle.load(open(metadata_path, 'rb'))
            print('Loading songs data from cache: {}. If you updated dataset remove {} before training!'.format(metadata_path, os.path.basename(metadata_path)))
        except Exception:
            print('Collecting metadata for', str(data_path), 'Dataset type:', self.dataset_type)
            if self.dataset_type in [1, 4]:
                metadata = []
                track_paths = []
                if type(data_path) == list:
                    for tp in data_path:
                        track_paths += sorted(glob(tp + '/*'))
                else:
                    track_paths += sorted(glob(data_path + '/*'))

                track_paths = [path for path in track_paths if os.path.basename(path)[0] != '.' and os.path.isdir(path)]
                for path in tqdm(track_paths):
                    length = len(sf.read(path + f'/{instruments[0]}.wav')[0])
                    metadata.append((path, length))
            elif self.dataset_type == 2:
                metadata = dict()
                for instr in self.instruments:
                    metadata[instr] = []
                    track_paths = []
                    if type(data_path) == list:
                        for tp in data_path:
                            track_paths += sorted(glob(tp + '/{}/*.wav'.format(instr)))
                    else:
                        track_paths += sorted(glob(data_path + '/{}/*.wav'.format(instr)))

                    for path in tqdm(track_paths):
                        length = len(sf.read(path)[0])
                        metadata[instr].append((path, length))
            elif self.dataset_type == 3:
                import pandas as pd
                if type(data_path) != list:
                    data_path = [data_path]

                metadata = dict()
                for i in range(len(data_path)):
                    print('Reading tracks from: {}'.format(data_path[i]))
                    df = pd.read_csv(data_path[i])

                    skipped = 0
                    for instr in self.instruments:
                        part = df[df['instrum'] == instr].copy()
                        print('Tracks found for {}: {}'.format(instr, len(part)))
                    for instr in self.instruments:
                        part = df[df['instrum'] == instr].copy()
                        metadata[instr] = []
                        track_paths = list(part['path'].values)
                        for path in tqdm(track_paths):
                            if not os.path.isfile(path):
                                print('Cant find track: {}'.format(path))
                                skipped += 1
                                continue
                            # print(path)
                            try:
                                length = len(sf.read(path)[0])
                            except:
                                print('Problem with path: {}'.format(path))
                                skipped += 1
                                continue
                            metadata[instr].append((path, length))
                    if skipped > 0:
                        print('Missing tracks: {} from {}'.format(skipped, len(df)))
            else:
                print('Unknown dataset type: {}. Must be 1, 2 or 3'.format(self.dataset_type))
                exit()

            pickle.dump(metadata, open(metadata_path, 'wb'))

        if self.dataset_type in [1, 4]:
            print('Found tracks in dataset: {}'.format(len(metadata)))
        else:
            for instr in self.instruments:
                print('Found tracks for {} in dataset: {}'.format(instr, len(metadata[instr])))
        self.metadata = metadata
        self.chunk_size = config.audio.chunk_size
        self.min_mean_abs = config.audio.min_mean_abs

    def __len__(self):
        return self.config.training.num_steps * self.config.training.batch_size

    def load_source(self, metadata, instr):
        while True:
            if self.dataset_type in [1, 4]:
                track_path, track_length = random.choice(metadata)
                source = load_chunk(track_path + f'/{instr}.wav', track_length, self.chunk_size)
            else:
                track_path, track_length = random.choice(metadata[instr])
                source = load_chunk(track_path, track_length, self.chunk_size)
            if np.abs(source).mean() >= self.min_mean_abs:  # remove quiet chunks
                break
        if self.aug:
            source = self.augm_data(source, instr)
        return torch.tensor(source, dtype=torch.float32)

    def load_random_mix(self):
        res = []
        for instr in self.instruments:
            # Multiple mix of sources
            s1 = self.load_source(self.metadata, instr)
            if self.config.training.augmentation_mix:
                if random.uniform(0, 1) < 0.2:
                    s2 = self.load_source(self.metadata, instr)
                    w1 = random.uniform(0.5, 1.5)
                    w2 = random.uniform(0.5, 1.5)
                    s1 = (w1 * s1 + w2 * s2) / (w1 + w2)
                    if random.uniform(0, 1) < 0.1:
                        s2 = self.load_source(self.metadata, instr)
                        w1 = random.uniform(0.5, 1.5)
                        w2 = random.uniform(0.5, 1.5)
                        s1 = (w1 * s1 + w2 * s2) / (w1 + w2)

            res.append(s1)
        res = torch.stack(res)
        return res

    def load_aligned_data(self):
        track_path, track_length = random.choice(self.metadata)
        res = []
        for i in self.instruments:
            attempts = 10
            while attempts:
                source = load_chunk(track_path + f'/{i}.wav', track_length, self.chunk_size)
                if np.abs(source).mean() >= self.min_mean_abs:  # remove quiet chunks
                    break
                attempts -= 1
                if attempts <= 0:
                    print('Attempts max!', track_path)
            res.append(source)
        res = np.stack(res, axis=0)
        if self.aug:
            for i, instr in enumerate(self.instruments):
                res[i] = self.augm_data(res[i], instr)
        return torch.tensor(res, dtype=torch.float32)

    def augm_data(self, source, instr):
        # source.shape = (2, 261120)

        
        # Channel shuffle
        if random.uniform(0, 1) < 0.25:
            source = source[::-1].copy()
        """
        # ALREDY IN SIMPLE AUGs or unused

        # Random inverse (do with low probability)
        if random.uniform(0, 1) < 0.01:
            source = source[:, ::-1].copy()

        # Random polarity (multiply -1)
        if random.uniform(0, 1) < 0.25:
            source = -source.copy()
        """

        if self.augment_func[instr]:
            source_init = source.copy()
            source = self.augment_func[instr](samples=source, sample_rate=44100)
            if source_init.shape != source.shape:
                source = source[..., :source_init.shape[-1]]

        # Random Reverb
        if random.uniform(0, 1) < 0.1:
            room_size = random.uniform(0.05, 1)
            damping = random.uniform(0, 1)
            wet_level = random.uniform(0.05, 1)
            dry_level = 1
            width = random.uniform(0, 1.0)
            board = PB.Pedalboard([PB.Reverb(
                room_size=room_size,
                damping=damping,
                wet_level=wet_level,
                dry_level=dry_level,
                width=width,
                freeze_mode=0,
            )])
            source = board(source, 44100)

        # Random Chorus
        if random.uniform(0, 1) < 0.05:
            rate_hz = random.uniform(0.2, 15.0)
            depth = random.uniform(0.25, 0.95)
            centre_delay_ms = random.uniform(3, 10)
            feedback = random.uniform(0.0, 0.5)
            mix = random.uniform(0.1, 0.5)
            board = PB.Pedalboard([PB.Chorus(
                rate_hz=rate_hz,
                depth=depth,
                centre_delay_ms=centre_delay_ms,
                feedback=feedback,
                mix=mix,
            )])
            source = board(source, 44100)

        # Random Phazer
        if random.uniform(0, 1) < 0.05:
            rate_hz = random.uniform(0.2, 15.0)
            depth = random.uniform(0.25, 0.95)
            centre_frequency_hz = random.uniform(200, 12000)
            feedback = random.uniform(0.0, 0.5)
            mix = random.uniform(0.1, 0.5)
            board = PB.Pedalboard([PB.Phaser(
                rate_hz=rate_hz,
                depth=depth,
                centre_frequency_hz=centre_frequency_hz,
                feedback=feedback,
                mix=mix,
            )])
            source = board(source, 44100)


        # Random Bitcrush
        if random.uniform(0, 1) < 0.05:
            bit_depth = random.uniform(3, 8)
            board = PB.Pedalboard([PB.Bitcrush(
                bit_depth=bit_depth
            )])
            source = board(source, 44100)

        
        """
        # UNUSED SINCE SAFETY LIMITER ADDED
        # Random Limiter
        if random.uniform(0, 1) < 0.2:
            board = PB.Pedalboard([PB.Limiter(
              threshold_db = random.uniform(-12, 0),
              release_ms = random.uniform(5.0, 200.0)
            )])
            source = board(source, 44100)

        # audiomentations tanh saturation is way better (better sounding, and gain compensated) !
        # this one is creating huge volume difference !
        # Random Distortion
        if random.uniform(0, 1) < 0.05:
            drive_db = random.uniform(1.0, 25.0)
            board = PB.Pedalboard([PB.Distortion(
                drive_db=drive_db,
            )])
            source = board(source, 44100)

        # TOO MUCH SLOW, audiomentations pitchshift is better and 2x faster
        # Random PitchShift
        if random.uniform(0, 1) < 0.33:
            semitones = random.uniform(-4, 4)
            board = PB.Pedalboard([PB.PitchShift(
                semitones=semitones
            )])
            source = board(source, 44100)
        
        # NOT USED
        # Random Resample
        if random.uniform(0, 1) < 0.05:
            target_sample_rate = random.uniform(4000, 44100)
            board = PB.Pedalboard([PB.Resample(
                target_sample_rate=target_sample_rate
            )])
            source = board(source, 44100)
        
        
        # ALREADY IN SIMPLE AUGs & UNUSED
        # Random MP3Compressor
        if random.uniform(0, 1) < 0.05:
            vbr_quality = random.uniform(0, 9.999)
            board = PB.Pedalboard([PB.MP3Compressor(
                vbr_quality=vbr_quality
            )])
            source = board(source, 44100)
        """ 


        # Parallel crush compressor
        if random.uniform(0, 1) < 0.1:
            board = PB.Pedalboard([PB.Compressor(
              threshold_db = -24,
              ratio = 4,
              attack_ms = 1,
              release_ms = 1
            )])
            source_comp = board(source, 44100)
            source = (4 * source + source_comp) / 5


        
        # Safety pre compressor
        peak = np.max(np.abs((source)))
        if peak > 0.9:
            board = PB.Pedalboard([PB.Compressor(
              threshold_db = -9,
              ratio = 2,
              attack_ms = 1,
              release_ms = 10
            )])
            source = board(source, 44100)

        # Safety Limiter
        peak = np.max(np.abs((source)))
        if peak > 1:
            board = PB.Pedalboard([PB.Limiter(
              threshold_db = 0,
              release_ms = random.uniform(5.0, 200.0)
            )])
            source = board(source, 44100) * 0.8 # lower volume for better balance with less limited stems

        return source


    def __getitem__(self, index):
        if self.dataset_type in [1, 2, 3]:
            res = self.load_random_mix()
        else:
            res = self.load_aligned_data()

        # Randomly change loudness of each stem
        if self.config.training.augmentation_loudness:
            if self.config.training.augmentation_loudness_type == 1:
                split = random.uniform(
                    self.config.training.augmentation_loudness_min,
                    self.config.training.augmentation_loudness_max
                )
                res[0] *= split
                res[1] *= (2 - split)
            else:
                for i in range(len(res)):
                    loud = random.uniform(
                        self.config.training.augmentation_loudness_min,
                        self.config.training.augmentation_loudness_max
                    )
                    res[i] *= loud
                    

        
        mix = res.sum(0)
        
        # Normalise created mixture to (-1, 1)
        # and apply same gain change on all stems.
        if 1:
            peakmix = torch.max(torch.abs(torch.tensor(mix)))
            mix /= peakmix
            for i in range(len(res)):
                res[i] /= peakmix
        
        ###
        ### "MASTERING" STAGE
        ### SIDECHAINED-LIMITER
        ### WOULD GO
        ### HERE 
        ###

        if self.mp3_aug is not None:
            mix = self.mp3_aug(samples=mix, sample_rate=44100)

        # If we need given stem (for roformers)
        if self.config.training.target_instrument is not None:
            index = self.config.training.instruments.index(self.config.training.target_instrument)
            return res[index], mix

        # sf.write("mix.wav", mix.T, 44100, subtype='PCM_16')
        return res, mix
