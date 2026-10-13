from scipy.io import loadmat


class EMGExplorer:

    def __init__(self, mat_file):

        self.mat_file = mat_file

        self.data = None

    # ==========================================
    # Load MAT File
    # ==========================================

    def load(self):

        print("=" * 60)
        print("Loading MAT File")
        print("=" * 60)

        self.data = loadmat(self.mat_file)

        print("Loaded Successfully")

    # ==========================================
    # Show Variables
    # ==========================================

    def show_variables(self):

        print()

        print("=" * 60)
        print("Variables Inside MAT")
        print("=" * 60)

        for key, value in self.data.items():

            if key.startswith("__"):
                continue

            print()

            print(f"Variable : {key}")

            print(f"Type     : {type(value)}")

            try:
                print(f"Shape    : {value.shape}")
            except:
                print("Shape    : N/A")