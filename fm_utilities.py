import librosa
import numpy as np
import scipy.signal as sps

# TODO: add time-varying f0 option (or make do it by default)
def extract_modulating_delay(sig,fs,width=0.05) :
    # extract f0 (not time-varying)
    f0s, flags, probs = librosa.pyin(s,fmin=librosa.note_to_hz('C2'),fmax=librosa.note_to_hz('C7'), sr=fs)
    filter_f0 = np.median(f0s[flags])
    w0 = (2*np.pi*filter_f0)/fs
    # bp filter signal and get analytic signal
    f_low = (1-width)*filter_f0
    f_high = (1+width)*filter_f0
    b,a = sps.butter(3, [f_low, f_high], btype='bandpass', fs=fs)
    filt_sig = sps.lfilter(b,a,sig)
    filt_sig_hilb = sps.hilbert(filt_sig)

    # find attack and release time
    sig_rms = librosa.feature.rms(y=sig)
    t = librosa.samples_like(sig_rms)
    sig_rms = sig_rms.flatten()
    full_rms = np.interp(np.arange(sig.shape[0]),t,sig_rms)
    amp_max = np.max(full_rms)
    attack_time = np.where(full_rms > amp_max * 0.1)[0][0]
    release_time = np.where(full_rms > amp_max * 0.1)[0][-1]

    # get delay
    wi = np.diff(np.unwrap(np.angle(filt_sig_hilb)))
    d_sig = np.zeros(sig.shape)
    d_sig[attack_time:release_time] = np.cumsum(1 - wi[attack_time:release_time]/w0)
    d_sig[release_time:] = d_sig[release_time-1]
    return d_sig

def fm_demodulation(sig,mod_delay,recursive=False) :
    demod_delay = -mod_delay
    delay_min = np.min(demod_delay)
    if delay_min < 0 :
        offset = int(-np.floor(delay_min))
        demod_delay += offset
        signal = np.hstack([np.zeros(offset),signal])
    new_signal = np.zeros(signal.shape[0], dtype=signal.dtype)
    if recursive :
        demod_delay = delay_line(demod_delay,demod_delay)
    return delay_line(new_signal,demod_delay)

def delay_line(sig,delay) :
    new_signal = np.zeros(sig)
    for i in range(delay.shape[0]) :
        # modulation is supposed to start after attack and end before release
        # so we should be safe in practice, but we should check indices here
        index = i - delay[i]

        # these can happen - need to think about the beginning and end
        # and the phase of the modulator
        if index < 0 :
            continue
        elif index >= signal.shape[0] - 1 :
            continue
        # start with linear interpolation, but we will need something better later
        frac = index - int(index)
        new_signal[i] = (1-frac)*signal[int(index)] + frac*signal[int(index)+1]
    return new_signal
