
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import sounddevice as sd
from IPython.display import display, Audio



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
        fig, axs = plt.subplots(1, num_plots, figsize=(19*num_plots, 11))
    else:
        fig, axs = plt.subplots(N, num_plots, figsize=(19*num_plots, 11*N))
    
    # Iterate over signals
    for i in range(N):
        plot_index = 0
        
        # Plot waveform
        if show_waveform:
            plt.subplot(N, num_plots, i * num_plots + 1)
            plt.plot(np.arange(len(signals_np[i][start:end])) / fs, signals_np[i][start:end])
            plt.title(f'Waveform - {names[i] if isinstance(names, list) else names} {i+1}')
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')
            plt.ylim(-1, 1)
            plt.grid(True)
            plot_index += 1
        
        # Plot spectrogram
        if show_spectrogram:
            plt.subplot(N, num_plots, i * num_plots + plot_index + 1)
            plt.specgram(signals_np[i][start:end], NFFT=1024, Fs=fs, cmap='viridis', vmin=-80)
            plt.title(f'Spectrogram - {names[i] if isinstance(names, list) else names} {i+1}')
            plt.xlabel('Time (s)')
            plt.ylabel('Frequency (Hz)')
            plt.ylim(0, fmax)
            plt.grid(True)
            plot_index += 1
        
        # Plot histogram
        if show_histogram:
            plt.subplot(N, num_plots, i * num_plots + plot_index + 1)
            plt.hist(signals_np[i][start:end], bins=50)
            plt.title(f'Histogram - {names[i] if isinstance(names, list) else names} {i+1}')
            plt.xlabel('Amplitude')
            plt.ylabel('Frequency')
            plt.xlim(-1, 1)
            plt.grid(True)
    
    plt.tight_layout()
    plt.show()


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



def audio_widget(audio_np, samplerate, name='audio'):
    """
    Create Audio widgets from a 2D NumPy array of audio sources.

    Parameters:
    - audio_np (numpy.ndarray): 2D array containing audio sources.
    - samplerate (int): Sample rate of the audio sources.
    - name (str or list): Name prefix for the audio widgets. If it's a list, each audio source will be named accordingly.
    """
    if isinstance(name, list):
        assert len(name) == audio_np.shape[0], "Number of names should match the number of audio sources."
        for i, audio in enumerate(audio_np):
            print(f'{name[i]}:')
            display(Audio(audio, rate=samplerate))
    else:
        for i, audio in enumerate(audio_np):
            print(f'{name} {i+1}:')
            display(Audio(audio, rate=samplerate))



































