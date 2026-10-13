import mne
import pywt
import numpy as np


class PreprocessingFilters:

    def __init__(self, raw):

        self.raw = raw.copy()

    # ==========================================================
    # Bandpass Filter
    # ==========================================================

    def bandpass(self, low=1.0, high=40.0):

        print(f"Applying Bandpass Filter ({low}-{high} Hz)...")

        raw = self.raw.copy()

        try:

            raw.filter(
                l_freq=low,
                h_freq=high,
                verbose=False
            )

            print("Bandpass completed.")

            return raw

        except Exception as e:

            print(f"Bandpass failed : {e}")

            return self.raw.copy()

    # ==========================================================
    # Notch Filter
    # ==========================================================

    def notch(self, freq=50):

        print(f"Applying Notch Filter ({freq} Hz)...")

        raw = self.raw.copy()

        try:

            raw.notch_filter(
                freqs=freq,
                verbose=False
            )

            print("Notch completed.")

            return raw

        except Exception as e:

            print(f"Notch failed : {e}")

            return self.raw.copy()

    # ==========================================================
    # ICA Artifact Removal
    # ==========================================================

    def ica(self, n_components=20):

        print("Applying ICA...")

        raw = self.raw.copy()

        try:

            ica = mne.preprocessing.ICA(
                n_components=n_components,
                random_state=42,
                max_iter="auto"
            )

            ica.fit(raw)

            raw = ica.apply(raw)

            print("ICA completed.")

            return raw

        except Exception as e:

            print(f"ICA failed : {e}")

            return self.raw.copy()

    # ==========================================================
    # Wavelet Denoising
    # ==========================================================

    def wavelet(self,
                 wavelet="db4",
                 level=4):

        print("Applying Wavelet Denoising...")

        raw = self.raw.copy()

        data = raw.get_data()

        cleaned = []

        try:

            for signal in data:

                coeffs = pywt.wavedec(
                    signal,
                    wavelet,
                    level=level
                )

                sigma = (
                    np.median(
                        np.abs(coeffs[-1])
                    ) / 0.6745
                )

                threshold = sigma * np.sqrt(
                    2 * np.log(len(signal))
                )

                # Threshold ONLY detail coefficients
                coeffs[1:] = [

                    pywt.threshold(
                        c,
                        threshold,
                        mode="soft"
                    )

                    for c in coeffs[1:]

                ]

                denoised = pywt.waverec(
                    coeffs,
                    wavelet
                )

                cleaned.append(
                    denoised[:len(signal)]
                )

            raw._data = np.asarray(cleaned)

            print("Wavelet completed.")

            return raw

        except Exception as e:

            print(f"Wavelet failed : {e}")

            return self.raw.copy()

    # ==========================================================
    # Channel Interpolation
    # ==========================================================

    def interpolate(self, bad_channels):

        print("Interpolating Bad Channels...")

        raw = self.raw.copy()

        try:

            raw.info["bads"] = bad_channels

            raw.interpolate_bads(
                reset_bads=True,
                verbose=False
            )

            print("Interpolation completed.")

            return raw

        except Exception as e:

            print(f"Interpolation failed : {e}")

            return self.raw.copy()