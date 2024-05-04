
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import sounddevice as sd
from IPython.display import display, Audio



# This function assumes as input a matrix X with samples on columns
def preprocessing(X, n_sources, under_complete = True):
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




# This function assumes as input a matrix X with samples on columns, automatically performs the preprocessing step
def FastIca(mixture_signals, n_sources, n_iter = 5000, tol = 1e-9, under_complete = True):
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


def norm_signals(signals_np, start=0, end=None): 
    """
    Normalize signals in a NumPy array by dividing each signal by its maximum absolute value.

    Parameters:
    - signals_np (numpy.ndarray): 1D or 2D array containing signals.
    - start (int): Start index for normalization (default: 0).
    - end (int): End index for normalization (default: None, which means until the end).

    Returns:
    - signals_normalized (numpy.ndarray): Normalized signals.
    """
    # If signals_np is 1D, convert it to a 2D array with one column
    if signals_np.ndim == 1:
        signals_np = signals_np[:, np.newaxis]

    # Calculate the maximum absolute value along the appropriate axis
    max_values = np.max(np.abs(signals_np), axis=1, keepdims=True)

    # Normalize the signals
    if end is None:
        signals_normalized = signals_np[:, start:] / max_values
    else:
        signals_normalized = signals_np[:, start:end] / max_values

    return signals_normalized



# Function to plot the signals
def plot_signals(t, S, title, labels = False, scaling = 1, xlim = 30, linewidth = 4): 
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

import numpy as np
import matplotlib.pyplot as plt



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
    # Convert 1D array to 2D with single row if needed
    if signals_np.ndim == 1:
        signals_np = np.expand_dims(signals_np, axis=0)

    # Number of signals
    N = signals_np.shape[0]
    # Number of plots to show
    num_plots = sum([show_waveform, show_spectrogram, show_histogram])

    # Set up subplots
    fig, axs = plt.subplots(nrows=N, ncols=num_plots, figsize=(11, 3*N))

    for i in range(N):
        col = 0  # Start column index for the subplot

        # Plot waveform
        if show_waveform:
            axs[i, col].plot(np.arange(len(signals_np[i][start:end])) / fs, signals_np[i][start:end])
            axs[i, col].set_title(f'Waveform - {names[i] if isinstance(names, list) else names} {i+1}')
            axs[i, col].set_xlabel('Time (s)')
            axs[i, col].set_ylabel('Amplitude')
            axs[i, col].set_ylim(-1, 1)  # Set y-axis limits
            axs[i, col].grid(True)  # Add gridlines
            col += 1

        # Plot spectrogram
        if show_spectrogram:
            axs[i, col].specgram(signals_np[i][start:end], NFFT=1024, Fs=fs, cmap='viridis', vmin=-80)
            axs[i, col].set_title(f'Spectrogram - {names[i] if isinstance(names, list) else names} {i+1}')
            axs[i, col].set_xlabel('Time (s)')
            axs[i, col].set_ylabel('Frequency (Hz)')
            axs[i, col].set_ylim(0, fmax)  # Set y-axis limits
            axs[i, col].grid(True)  # Add gridlines
            col += 1

        # Plot histogram
        if show_histogram:
            axs[i, col].hist(signals_np[i][start:end], bins=50)
            axs[i, col].set_title(f'Histogram - {names[i] if isinstance(names, list) else names} {i+1}')
            axs[i, col].set_xlabel('Amplitude')
            axs[i, col].set_ylabel('Frequency')
            axs[i, col].set_xlim(-1, 1)  # Set x-axis limits
            axs[i, col].grid(True)  # Add gridlines

    plt.tight_layout()
    plt.show()





def play_audio_from_array(audio_np, samplerate=44100):
    # Play the audio
    sd.play(audio_np, samplerate=samplerate)

    # Use sd.wait() to wait until file is done playing
    sd.wait()


def audio_widget(audio_np, samplerate, name='audio'):
    """
    Create Audio widgets from a 2D NumPy array of audio sources.

    Parameters:
    - audio_np (numpy.ndarray): 2D array containing audio sources.
    - samplerate (int): Sample rate of the audio sources.
    - name (str): Name prefix for the audio widgets (default: 'audio').
    """
    for i in range(audio_np.shape[0]):
        print(f'{name} {i+1}:')
        display(Audio(audio_np[i, :], rate=samplerate))


































