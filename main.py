def main():
    from src.pipline.training_pipeline import TrainPipeline

    train_pipeline = TrainPipeline()
    train_pipeline.run_pipeline()


if __name__ == "__main__":
    main()
