from mil_toolbox.data import DummyWSIDataset

def main():
    dataset = DummyWSIDataset(num_wsi=100)
    print("Dataset created:")

    for i in range(len(dataset)):
        wsi, label = dataset[i]
        print(f"WSI {i}: {wsi.shape}, Label {i}: {label}")

if __name__ == "__main__":
    main()
