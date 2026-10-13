class LabelMapper:
    """
    Converts dataset-specific labels into one
    common movement vocabulary used by the
    Adaptive Fusion Engine.
    """

    def __init__(self):

        # -----------------------------
        # PhysioNet EEG Mapping
        # -----------------------------

        self.physionet_mapping = {

            1: {
                "T0": "REST"
            },

            2: {
                "T0": "REST"
            },

            3: {
                "T0": "REST",
                "T1": "LEFT_HAND_CLOSE",
                "T2": "RIGHT_HAND_CLOSE"
            },

            4: {
                "T0": "REST",
                "T1": "LEFT_HAND_IMAGERY",
                "T2": "RIGHT_HAND_IMAGERY"
            },

            5: {
                "T0": "REST",
                "T1": "BOTH_HANDS_CLOSE",
                "T2": "BOTH_FEET_MOVE"
            },

            6: {
                "T0": "REST",
                "T1": "BOTH_HANDS_IMAGERY",
                "T2": "BOTH_FEET_IMAGERY"
            },

            7: {
                "T0": "REST",
                "T1": "LEFT_HAND_CLOSE",
                "T2": "RIGHT_HAND_CLOSE"
            },

            8: {
                "T0": "REST",
                "T1": "LEFT_HAND_IMAGERY",
                "T2": "RIGHT_HAND_IMAGERY"
            },

            9: {
                "T0": "REST",
                "T1": "BOTH_HANDS_CLOSE",
                "T2": "BOTH_FEET_MOVE"
            },

            10: {
                "T0": "REST",
                "T1": "BOTH_HANDS_IMAGERY",
                "T2": "BOTH_FEET_IMAGERY"
            },

            11: {
                "T0": "REST",
                "T1": "LEFT_HAND_CLOSE",
                "T2": "RIGHT_HAND_CLOSE"
            },

            12: {
                "T0": "REST",
                "T1": "LEFT_HAND_IMAGERY",
                "T2": "RIGHT_HAND_IMAGERY"
            },

            13: {
                "T0": "REST",
                "T1": "BOTH_HANDS_CLOSE",
                "T2": "BOTH_FEET_MOVE"
            },

            14: {
                "T0": "REST",
                "T1": "BOTH_HANDS_IMAGERY",
                "T2": "BOTH_FEET_IMAGERY"
            }

        }

    # =======================================================
    # Convert PhysioNet label
    # =======================================================

    def map_physionet(self,
                      run_number,
                      label):

        run_mapping = self.physionet_mapping.get(run_number)

        if run_mapping is None:

            return "UNKNOWN"

        return run_mapping.get(label, "UNKNOWN")