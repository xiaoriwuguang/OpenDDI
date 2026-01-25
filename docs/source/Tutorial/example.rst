Example
========

.. code-block:: python

    import torch
    from openddi.utils.logger import create_logger
    from openddi.parms_setting import settings, set_random_seed
    from openddi.data.dataset_manager import dataset_manager
    from openddi.models.model_manager import model_manager
    from openddi.pipeline import Pipeline

    # Set random seed for reproducibility
    set_random_seed(1, deterministic=True)
    
    # Parse command-line arguments
    args = settings()
    
    # Create logger
    logger = create_logger(args)
    
    # Set device
    args.cuda = (args.device == 'cuda')

    # Load dataset
    Dataset_manager = dataset_manager(args)
    ddi_dataset = Dataset_manager.load_dataset()
    ddi_dataset.load_data()

    # Load model
    Model_manager = model_manager(args)
    if args.origin:
        model = Model_manager.load_origin_model(ddi_dataset)
    else:
        model = Model_manager.load_model()
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    # Create and run pipeline
    ddi_pipeline = Pipeline(args, logger, ddi_dataset, model, optimizer)
    ddi_pipeline.run()