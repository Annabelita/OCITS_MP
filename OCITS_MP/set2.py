from training_complex import run


def main():
    training_styles = ["3GPU_async"]
    for training_style in training_styles:
        run(training_style)


if __name__ == "__main__":
    main()
