

if config.setup.stage != 0:

    print('Generating predictions...')

    submission, all_classes_preds = test_inference(test_loader, best_model)

    print('Number of unique sirnas', submission['predicted_sirna'].nunique())

    save_csv(config, submission, all_classes_preds)
