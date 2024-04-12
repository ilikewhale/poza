import torch
import torch.nn as nn
import torch.nn.functional as F
import re_layer
import numpy as np
import soundfile as sf

class NSynthEncoder(nn.Module):
    def __init__(self, input_channels, ae_width, ae_filter_length, ae_num_layers, ae_num_stages, ae_bottleneck_width, rescale_inputs=False):
        super(NSynthEncoder, self).__init__()
        self.rescale_inputs = rescale_inputs
        self.input_channels = input_channels
        self.ae_width = ae_width
        self.ae_filter_length = ae_filter_length
        self.ae_num_layers = ae_num_layers
        self.ae_num_stages = ae_num_stages
        self.ae_bottleneck_width = ae_bottleneck_width
        
        # Start convolution
        self.start_conv = re_layer.Conv1d(input_channels, ae_width, ae_filter_length, padding=ae_filter_length // 2)
        
        # Dilated convolutions
        self.dilated_convs = nn.ModuleList([
            re_layer.dilated_conv1d(ae_width, ae_width, ae_filter_length, padding=(ae_filter_length // 2) * (2**(layer % ae_num_stages)), dilation=2**(layer % ae_num_stages))
            for layer in range(ae_num_layers)
        ])
        
        # Residual connections
        self.res_convs = nn.ModuleList([
            re_layer.Conv1d(ae_width, ae_width, 1)
            for _ in range(ae_num_layers)
        ])
        
        # Bottleneck convolution
        self.bottleneck_conv = re_layer.Conv1d(ae_width, ae_bottleneck_width, 1)

    def forward(self, x):
        if self.rescale_inputs:
            x = x * 2 - 1  # Rescale input if necessary
        
        x = F.relu(self.start_conv(x))
        for dilated_conv, res_conv in zip(self.dilated_convs, self.res_convs):
            d = F.relu(dilated_conv(x))
            x += res_conv(d)
        
        x = self.bottleneck_conv(x)
        return x


class AudioGenerator:
    def __init__(self, model, sample_rate=16000, samples_per_save=10000):
        """
        Initializes the audio generator with a pre-trained model.

        Args:
          model: A pre-trained PyTorch model to generate audio from encodings.
          sample_rate: The sample rate of the audio to generate.
          samples_per_save: Number of samples to process before saving audio to disk.
        """
        self.model = model
        self.sample_rate = sample_rate
        self.samples_per_save = samples_per_save
        self.model.eval()  # Set the model to evaluation mode

    def synthesize(self, encodings1, encodings2, save_paths):
        """
        Generate and save audio from a batch of encodings.

        Args:
        encodings1: Numpy array with shape [batch_size, time, dim] for the first instrument.
        encodings2: Numpy array with shape [batch_size, time, dim] for the second instrument.
        save_paths: Iterable of strings, paths to save generated audio files.
        """
        with torch.no_grad():  # Disable gradient computation
            batch_size, encoding_length, _ = encodings1.shape
            hop_length = self.model.hop_length
            total_length = encoding_length * hop_length

            # Initialize an array to hold the generated audio samples
            audio_batch = np.zeros((batch_size, total_length), dtype=np.float32)

            # Generate audio sample by sample
            for sample_i in range(total_length):
                encoding_i = sample_i // hop_length
                encoding_batch = torch.tensor((encodings1[:, encoding_i, :] + encodings2[:, encoding_i, :]) / 2, dtype=torch.float32)
                audio_sample = self.model.generate(encoding_batch)  # Model generates audio for the current encoding

                audio_batch[:, sample_i] = audio_sample.cpu().numpy()[:, 0]

                if sample_i % self.samples_per_save == 0 and sample_i > 0:
                    # Save the generated audio to file
                    self.save_batch(audio_batch, sample_i, save_paths)

            # Final save for any remaining audio samples
            self.save_batch(audio_batch, None, save_paths)


    def save_batch(self, audio_batch, sample_index, save_paths):
        """
        Save the batch of audio data to files.

        Args:
          audio_batch: Numpy array containing the audio samples.
          sample_index: Current index in the sample generation, or None if final save.
          save_paths: Iterable of strings, paths to save generated audio files.
        """
        if sample_index is not None:
            for i, path in enumerate(save_paths):
                sf.write(path, audio_batch[i, :sample_index], self.sample_rate)
                print(f"Audio saved at sample {sample_index} to {path}")
        else:
            for i, path in enumerate(save_paths):
                sf.write(path, audio_batch[i], self.sample_rate)
                print(f"Final audio saved to {path}")

