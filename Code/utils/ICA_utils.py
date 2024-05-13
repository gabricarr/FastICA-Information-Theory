import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import sounddevice as sd
from IPython.display import display, Audio
import pyroomacoustics as pra
import mir_eval


def preprocessing(X, n_sources, under_complete = True):
    """
    Preprocesses the input data matrix X for independent component analysis (ICA).

    Parameters:
    - X (numpy.ndarray): Input data matrix with samples on columns.
    - n_sources (int): Number of independent sources.
    - under_complete (bool): If True, performs under-complete ICA by reducing dimensionality to n_sources (default: True).

    Returns:
    - T_norm (numpy.ndarray): Preprocessed data matrix for ICA.
    """
    # Centering step 
    X_mean = np.mean(X, axis = 1)               # Samples on columns
    X_bar = X - X_mean[:, None]
 

    # PCA Dimensionality reduction
    U, s, VT = np.linalg.svd(X_bar, full_matrices = False)

    # If the number of sources is less than the number of mixtures we perform dimensionality reduction
    # and keep only the first n_sources principal components (under-complete ICA)
    if under_complete:
        if (n_sources < X.shape[0]):
            T = U[:, :n_sources].T @ X_bar              # First 'n_sources' Principal components    
            print("Dimensionality reduction performed")
        else:
            T = U.T @ X_bar
    else:
        T = U.T @ X_bar                     


    # Scaling / Normalization step
    T_std = np.std(T, axis = 1) 
    T_norm = (T) / T_std[:, None]

    return T_norm


def FastIca(mixture_signals, n_sources, n_iter = 5000, tol = 1e-9, under_complete = True):
    """
    Perform Independent Component Analysis (ICA) on the input mixture signals, automatically performs the preprocessing step.

    Parameters:
    - mixture_signals (numpy.ndarray): Input matrix with samples on columns.
    - n_sources (int): Number of independent sources.
    - n_iter (int): Maximum number of iterations for FastICA algorithm (default: 5000).
    - tol (float): Tolerance to declare convergence (default: 1e-9).
    - under_complete (bool): If True, performs under-complete ICA by reducing dimensionality to n_sources (default: True).

    Returns:
    - S (numpy.ndarray): Recovered signals.
    - W (numpy.ndarray): Unmixing matrix.
    """
    # Preprocessing step
    X = preprocessing(mixture_signals, n_sources, under_complete = under_complete)

    # Number of mixtures
    n_mixtures = X.shape[0]
    # Number of samples
    n_samples = X.shape[1]

    # Initialize a random weight Matrix with unit varinace
    W = np.random.rand(n_mixtures, n_sources) 
    W_std = np.std(W, axis = 1) 
    W = (W) / W_std[:, None]


    # FastICA algorithm
    for p in range(n_sources):
        w_p = W[:, p]                           # recover the p-th row of W

        # Computing a single independent component
        for i in range(n_iter):                 
            w_p_old = w_p

            y = w_p.T @ X                       # Argument of g
            # Computing the fisrt derivative  
            g = y*np.exp(-y**2/2)   
            # g = np.tanh(y)       

            # Computing the second derivative
            g_prime = (1 - y**2)*np.exp(-y**2/2)
            # g_prime = 1 - np.tanh(y)**2

            # Update rule
            w_p = X @ g.T / n_samples - np.mean(g_prime)*w_p

            # Orthogonalization step
            w_p = w_p - W[:, :p] @ W[:, :p].T @ w_p

            # Method 2 for orthogonalization
            # for k in range(p):
            #      w_p = w_p - np.dot(W[:, k], w_p) * W[:, k]

            # Normalization step
            w_p = w_p / np.linalg.norm(w_p)

            # Convergence criterion
            if (np.linalg.norm(w_p - w_p_old) < tol):
                break
        
        W[:, p] = w_p
    
    # Recovering the sources
    S = W.T @ X
    return S, W     


def import_audio_folder(folder_path, duration=None):
    """
    Import audio files from a specified folder, normalize them, and ensure they have the same duration and sampling frequency.

    Parameters:
    - folder_path (str): Path to the folder containing audio files.
    - duration (float): Desired duration in seconds for all samples (default: None).

    Returns:
    - processed_audio_data (numpy.ndarray): Normalized audio samples in a numpy array.
    - sampling_frequency (int): Sampling frequency of the audio files.
    - file_names (list): List of file names in the folder.
    - length (samples): sample length of all audio samples after the importing and trimming.
    """

    # Suppressione specifica delle avvertenze WavFileWarning
    warnings.filterwarnings("ignore", category=wavfile.WavFileWarning)

    # Find all the audio files in the folder
    file_names = [file for file in os.listdir(folder_path) if file.endswith(".wav")]

    # Initialize variables to store audio data and sampling frequencies
    audio_data = []
    sampling_frequencies = []

    # Iterate over each audio file
    for audio_file in file_names:
        # Load the file path for the current audio file
        file_path = os.path.join(folder_path, audio_file)

        # Read the audio file
        sampling_frequency, audio = wavfile.read(file_path)

        # Check if duration is specified and trim audio if necessary
        if duration is not None and len(audio) > duration * sampling_frequency:
            audio = audio[:int(duration * sampling_frequency)]

        # Normalize the audio sample
        audio = audio / np.max(np.abs(audio))

        # Store audio data and sampling frequency
        audio_data.append(audio)
        sampling_frequencies.append(sampling_frequency)

    # Check if all files have the same sampling frequency
    if len(set(sampling_frequencies)) != 1:
        raise ValueError("Not all files have the same sampling frequency.")

    # Ensure all audio samples have the same length
    length = min(len(audio) for audio in audio_data)
    processed_audio_data = np.array([audio[:length] for audio in audio_data])

    # Remove ".wav" extension from names
    file_names = [name.replace('.wav', '') for name in file_names]

    return processed_audio_data, sampling_frequencies[0], file_names, length


def norm_signals(signals_np, start=0, end=None, epsilon=1e-10): 
    """
    Normalize signals in a NumPy array by dividing each signal by its maximum absolute value.

    Parameters:
    - signals_np (numpy.ndarray): 1D or 2D array containing signals.
    - start (int): Start index for normalization (default: 0).
    - end (int): End index for normalization (default: None, which means until the end).
    - epsilon (float): Small value to avoid division by zero (default: 1e-10).

    Returns:
    - signals_normalized (numpy.ndarray): Normalized signals.
    """
    # If signals_np is 1D, convert it to a 2D array with one column
    if signals_np.ndim == 1:
        signals_np = signals_np[np.newaxis,:]

    # Calculate the maximum absolute value along the appropriate axis
    max_values = np.max(np.abs(signals_np), axis=1, keepdims=True)

    # Avoid division by zero
    max_values[max_values < epsilon] = epsilon

    # Normalize the signals
    if end is None:
        signals_normalized = signals_np[:, start:] / max_values
    else:
        signals_normalized = signals_np[:, start:end] / max_values

    return signals_normalized


def plot_signals(t, S, title, labels = False, scaling = 1, xlim = 30, linewidth = 4): 
    """
    Plot the signals superimposed  on the same plot.

    Parameters:
    - t (numpy.ndarray): Time axis.
    - S (numpy.ndarray): Matrix of signals to plot.
    - title (str): Title of the plot.
    - labels (bool): If True, display labels for each signal (default: False).
    - scaling (float): Scaling factor for the y-axis (default: 1).
    - xlim (int): Limit for the x-axis (default: 30).
    - linewidth (int): Width of the lines in the plot (default: 4).
    """
    # Creating the figure
    plt.figure(figsize=(20, 6))

    # Plotting the signals
    if labels:
        for i in range(S.shape[0]):
            plt.plot(t, S[i], label = "s "+str(i+1), linewidth = linewidth)
    else:
        for i in range(S.shape[0]):
            plt.plot(t, S[i], linewidth = linewidth)

    plt.title(title, fontsize = 20)
    plt.xlabel("t")
    plt.ylabel("Amplitude")

    # Setting limits for the axes
    plt.xlim(0, xlim)
    plt.ylim(-1*scaling, 1*scaling)

    if labels:
        plt.legend()


def plot_signals_sequential(t, X, type, scaling = 1, xlim = 30, linewidth = 4):
    """
    Plot the mixed signals sequentially.

    Parameters:
    - t (numpy.ndarray): Time axis.
    - X (numpy.ndarray): Matrix of mixed signals.
    - type (str): Type of the signals (e.g., "Mixed").
    - scaling (float): Scaling factor for the y-axis (default: 1).
    - xlim (int): Limit for the x-axis (default: 30).
    - linewidth (int): Width of the lines in the plot (default: 4).
    """
    n_mixtures = X.shape[0]

    # Plotting the mixed signals
    fig, axs = plt.subplots(nrows=n_mixtures, ncols=1, figsize = (20, 5*n_mixtures))
    axs = axs.flatten()

    for i in range(n_mixtures):
        axs[i].plot(t, X[i], linewidth = linewidth)     
        axs[i].set_title(type + " " + str(i+1), fontsize = 20)
        axs[i].set_xlabel("t")
        axs[i].set_ylabel("Amplitude")


        # Setting limits for the axes
        axs[i].set_xlim(0, xlim)
        axs[i].set_ylim(-1*scaling, 1*scaling)


def complete_plot(signals_np, fs, names='Signal', start=0, end=None, show_waveform=True, show_spectrogram=True, show_histogram=True, fmax=5000):
    """
    Plot waveforms, spectrograms, and histograms of signals.

    Parameters:
        signals_np (numpy.ndarray): 1D or 2D array containing signals.
        fs (float): Sampling frequency of the signals.
        names (list or str): List of names for each signal (default: 'Signal').
        start (int): Start index for plotting (default: 0).
        end (int): End index for plotting (default: None, which means until the end).
        show_waveform (bool): Whether to plot waveforms (default: True).
        show_spectrogram (bool): Whether to plot spectrograms (default: True).
        show_histogram (bool): Whether to plot histograms (default: True).
        fmax (int): Maximum frequency limit for spectrogram plotting (default: 5000 Hz).
    """
    # Suppress specific warning
    warnings.filterwarnings("ignore", message="divide by zero encountered in log10")

    # Set the font size
    plt.rcParams.update({'font.size': 18})

    # Ensure signals_np is 2D
    signals_np = np.atleast_2d(signals_np)
    
    # Number of signals
    N = signals_np.shape[0]
    
    # Calculate number of plots
    num_plots = sum([show_waveform, show_spectrogram, show_histogram])
    
    # Set up subplots
    if N == 1:
        fig, axs = plt.subplots(1, num_plots, figsize=(18*num_plots, 8))
    else:
        fig, axs = plt.subplots(N, num_plots, figsize=(18*num_plots, 8*N))
    
    # Iterate over signals
    for i in range(N):
        plot_index = 0
        
        # Plot waveform
        if show_waveform:
            plt.subplot(N, num_plots, i * num_plots + 1)
            plt.plot(np.arange(len(signals_np[i][start:end])) / fs, signals_np[i][start:end])
            plt.title(f'WAVEFORM - {names[i] if isinstance(names, list) else names} {i+1 if not isinstance(names, list) else ""}')
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')
            plt.ylim(-1, 1)
            plt.grid(True)
            plot_index += 1
        
        # Plot spectrogram
        if show_spectrogram:
            plt.subplot(N, num_plots, i * num_plots + plot_index + 1)
            epsilon = 1e-10
            plt.specgram(signals_np[i][start:end]+epsilon, NFFT=1024, Fs=fs, cmap='viridis', vmin=-80)
            plt.title(f'SPECTROGRAM - {names[i] if isinstance(names, list) else names} {i+1 if not isinstance(names, list) else ""}')
            plt.xlabel('Time (s)')
            plt.ylabel('Frequency (Hz)')
            plt.ylim(0, fmax)
            plt.grid(True)
            plot_index += 1
        
        # Plot histogram
        if show_histogram:
            plt.subplot(N, num_plots, i * num_plots + plot_index + 1)
            plt.hist(signals_np[i][start:end], bins=50)
            plt.title(f'HISTOGRAM - {names[i] if isinstance(names, list) else names} {i+1 if not isinstance(names, list) else ""}')
            plt.xlabel('Amplitude')
            plt.ylabel('Frequency')
            plt.xlim(-1, 1)
            plt.grid(True)
    
    plt.tight_layout()
    plt.show()


def plot_overlapped_signals(original_signals, reconstructed_signals, names, fs):
    """
    Plot overlapped original and reconstructed signals.

    Parameters:
        original_signals (list of arrays): List containing original signals.
        reconstructed_signals (list of arrays): List containing reconstructed signals.
        names (list of str): List of names for each signal.
        fs (float): Sampling frequency.
    """
    # Calculate time axis in seconds
    time_sec = np.arange(len(original_signals[0])) / fs

    # Create subplots for each pair of signals
    fig, axs = plt.subplots(nrows=len(original_signals), ncols=1, figsize=(10, 3 * len(original_signals)),
                            sharex=True, sharey=True)

    # Plot original and reconstructed signals in each subplot
    for i, (original, reconstructed) in enumerate(zip(original_signals, reconstructed_signals)):
        axs[i].plot(time_sec, original, label="Original", color='blue', alpha=0.8)  # Plot original signal
        axs[i].plot(time_sec, reconstructed, label="Reconstructed", color='orange', alpha=0.8)  # Plot reconstructed signal
        axs[i].set_title(f"{names[i]}")  # Set title for the subplot
        axs[i].grid(True)  # Add grid to the subplot
        axs[i].tick_params(axis='both', which='major', labelsize=8)  # Lower font size of major ticks
        axs[i].tick_params(axis='both', which='minor', labelsize=4)  # Lower font size of minor ticks
        axs[i].set_xlabel('Time [s]', fontsize=10)  # Lower font size of x-axis label
        axs[i].set_ylabel('Amplitude', fontsize=10)  # Lower font size of y-axis label

    # Create a single legend for all subplots
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')

    plt.suptitle('Overlapped Signals')  # Overall title for the plot
    plt.tight_layout()  # Adjust layout to prevent overlapping of subplots
    plt.show()  # Display the plot


def postprocessing(S, S_recovered):
    """
    Function to permute the reconstructed signals from fastICA in the correct order 
    so that they can be subsequently compared with the original signals.
    Additionally, it flips the signals if the correlation coefficient between the original 
    and the recovered signals is negative.

    Parameters:
    - S: numpy array, shape (n_sources, n_samples)
        Original sources.
    - S_recovered: numpy array, shape (n_sources, n_samples)
        Recovered sources.

    Returns:
    - S_recovered_perm: numpy array, shape (n_sources, n_samples)
        Permuted separated sources based on the permutation variable extracted by bss_eval
    """

    # Using mir_eval library to calculate BSS evaluation metrics
    _, _, _, perm = mir_eval.separation.bss_eval_sources(S, S_recovered, compute_permutation=True)

    # Permuting the separated sources based on the permutation variable
    # This step is crucial because the order of the separated_sources may not match the order of the original sources after fastICA 
    # We align them with the original sources to facilitate comparison and evaluation

    # Initialize an array to store the permuted separated sources
    S_recovered_perm = np.empty_like(S_recovered)

    # Permute the separated sources based on the permutation variable
    for i, p in enumerate(perm):
        S_recovered_perm[i] = S_recovered[p]

    # Now separated_sources_permuted contains the separated sources in the same order as the original sources

    # Adjust the sign of the recovered sources if necessary based on the correlation coefficient
    # If the correlation coefficient between the original and recovered signals is negative,
    # it indicates a phase inversion, so we flip the sign of the recovered signal to match.
    # Adjust the sign of the recovered sources if necessary based on the difference between maximum and minimum values
    for i in range(len(S)):
        # Calculate the maximum and minimum values of the original and recovered signals
        original_max = np.max(S[i])
        original_min = np.min(S[i])
        recovered_max = np.max(S_recovered_perm[i])
        recovered_min = np.min(S_recovered_perm[i])

        # Calculate the absolute differences
        max_max_diff = abs(abs(original_max) - abs(recovered_max))
        max_min_diff = abs(abs(original_max) - abs(recovered_min))

        # If the difference in the recovered signal is less than the difference in the original signal, flip its sign
        if max_max_diff > max_min_diff:
            S_recovered_perm[i] = -1 * S_recovered_perm[i]

    return S_recovered_perm


def plot_scores(names, bss_score):
    """
    Plots the Source Separation Metrics (SDR, SIR, SAR) and mean scores, and prints the mean scores along with the best-reconstructed signal.
    
    Parameters:
        names (list): List of names of the signals.
        bss_score (list): List containing three arrays of scores for SDR, SIR, and SAR respectively.
    """
    # Extract SDR, SIR, SAR from bss_score
    SDR = bss_score[0]
    SIR = bss_score[1]
    SAR = bss_score[2]
    
    # Calculate mean scores
    mean_scores = (SDR + SIR + SAR) / 3
    
    # Plot SDR, SIR, SAR
    plt.figure(figsize=(10, 6))
    plt.barh(names, SDR, color='skyblue', label='SDR')
    plt.barh(names, SIR, color='salmon', left=SDR, label='SIR')
    plt.barh(names, SAR, color='lightgreen', left=[i+j for i,j in zip(SDR, SIR)], label='SAR')
    plt.xlabel('Scores')
    plt.title('Source Separation Metrics')
    plt.legend()
    plt.grid(axis='x')
    plt.show()

    # Print the mean scores and the best-reconstructed signal
    print("Mean Scores:")
    for i, score in enumerate(mean_scores):
        print(f"{names[i]}: {score:.2f}")

    # Find the index of the maximum mean score
    best_signal_index = np.argmax(mean_scores)
    print(f"\nThe best-reconstructed signal is {names[best_signal_index]} with a mean score of {mean_scores[best_signal_index]:.2f}")


def play_audio_from_array(audio_np, samplerate=44100):
    """
    Play audio from a NumPy array.

    Parameters:
    - audio_np (numpy.ndarray): Audio data as a 1D NumPy array.
    - samplerate (int): Sampling rate of the audio (default: 44100).
    """
    # Play the audio
    sd.play(audio_np, samplerate=samplerate)

    # Use sd.wait() to wait until file is done playing
    sd.wait()


def audio_widget(audio_np, samplerate, names='audio'):
    """
    Create Audio widgets from a 2D NumPy array of audio sources.

    Parameters:
    - audio_np (numpy.ndarray): 2D array containing audio sources.
    - samplerate (int): Sample rate of the audio sources.
    - name (str or list): Name prefix for the audio widgets. If it's a list, each audio source will be named accordingly.
    """
    if isinstance(names, list):
        assert len(names) == audio_np.shape[0], "Number of names should match the number of audio sources."
        for i, audio in enumerate(audio_np):
            print(f'{names[i]}:')
            display(Audio(audio, rate=samplerate))
    else:
        for i, audio in enumerate(audio_np):
            print(f'{names} {i+1}:')
            display(Audio(audio, rate=samplerate))


def create_directional_object(mic_type, azimuth):
    """
    Create a directional object based on the microphone type and azimuth angle.

    Parameters:
        mic_type (str): Type of microphone pattern.
            Should be one of 'FIGURE_EIGHT', 'HYPERCARDIOID', 'CARDIOID', 'SUBCARDIOID', or 'OMNI'.
        azimuth (float): Azimuth angle (horizontal angle) at which the directional object should be oriented, in degrees.

    Returns:
        pra.CardioidFamily: Directional object representing the specified microphone pattern oriented at the given azimuth angle.
    """
    # Mapping of mic types to corresponding directivity patterns
    mic_type_mapping = {
        "FIGURE_EIGHT": pra.DirectivityPattern.FIGURE_EIGHT,
        "HYPERCARDIOID": pra.DirectivityPattern.HYPERCARDIOID,
        "CARDIOID": pra.DirectivityPattern.CARDIOID,
        "SUBCARDIOID": pra.DirectivityPattern.SUBCARDIOID,
        "OMNI": pra.DirectivityPattern.OMNI
    }
    
    # Check if the mic_type is valid
    if mic_type not in mic_type_mapping:
        raise ValueError("Invalid mic_type. Please choose one of FIGURE_EIGHT, HYPERCARDIOID, CARDIOID, SUBCARDIOID, or OMNI.")
    
    # Create the directional object using the CardioidFamily class
    directional_object = pra.CardioidFamily(
        orientation=pra.DirectionVector(azimuth=azimuth, degrees=True),  # Specify orientation
        pattern_enum=mic_type_mapping[mic_type]  # Specify directivity pattern
    )
    
    return directional_object







































