from relighting_data_reconstructor import relighting_reconstructor










if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument('--training_stage_from', type=int, default=2, choices=[1, 2, 3],
                        help="1: Geometry training, 2: BRDF initial, 3: Fine-tuning")
    parser.add_argument('--stage1_checkpoint', type=str, default=None,
                        help="Checkpoint for stage 1 to continue training")
    parser.add_argument('--stage2_checkpoint', type=str, default=None,
                        help="Checkpoint for stage 2 to continue training")
    parser.add_argument('--stage1_iterations', type=int, default=30000,
                        help="Iterations for stage 1")
    parser.add_argument('--stage2_iterations', type=int, default=60000,
                        help="Iterations for stage 2")
    parser.add_argument('--stage3_iterations', type=int, default=10000,
                        help="Iterations for stage 3")




    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=7000)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--interval", type=int, default=30_000,
                        help="A shared iteration interval for test and saving results and checkpoints.")
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])
    if args.training_stage_from == 1:
        args.iterations = args.iterations

    elif args.training_stage_from == 2:
        args.iterations = args.brdf_iterations
        # args.source_path = args.relighting_path
    else:  # stage 3
        args.iterations = args.stage3_iterations

    if args.interval > args.brdf_iterations:
        args.interval = args.brdf_iterations // 5

    if len(args.test_iterations) == 0:
        args.test_iterations.extend(list(range(args.interval, args.iterations + 1, args.interval)))
    if len(args.save_iterations) == 0:
        args.save_iterations.extend(list(range(args.interval, args.iterations + 1, args.interval)))
    if len(args.checkpoint_iterations) == 0:
        args.checkpoint_iterations.extend(list(range(args.interval, args.iterations + 1, args.interval)))

    print("Optimizing " + args.model_path)
    print(f"Training Stage: {args.training_stage_from}")

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # # Print parameters
    # print("\n" + "=" * 80)
    # print("TRAINING CONFIGURATION")
    # print("=" * 80)
    #
    # model_args = lp.extract(args)
    # opt_args = op.extract(args)
    # pipe_args = pp.extract(args)
    #
    # print("\n=== Model Parameters ===")
    # for attr in sorted(dir(model_args)):
    #     if not attr.startswith('_'):
    #         value = getattr(model_args, attr)
    #         print(f"  {attr:30} : {value}")
    #
    # print("\n=== Optimization Parameters ===")
    # for attr in sorted(dir(opt_args)):
    #     if not attr.startswith('_'):
    #         value = getattr(opt_args, attr)
    #         print(f"  {attr:30} : {value}")
    #
    # print("\n=== Pipeline Parameters ===")
    # for attr in sorted(dir(pipe_args)):
    #     if not attr.startswith('_'):
    #         value = getattr(pipe_args, attr)
    #         print(f"  {attr:30} : {value}")
    #
    # print("\n=== Stage Parameters ===")
    # stage_params = ['training_stage', 'stage1_iterations', 'stage2_iterations', 'stage3_iterations',
    #                 'stage1_checkpoint', 'stage2_checkpoint']
    # for param in stage_params:
    #     if hasattr(args, param):
    #         value = getattr(args, param)
    #         print(f"  {param:30} : {value}")
    #
    # print("\n=== Other Parameters ===")
    # other_params = ['ip', 'port', 'debug_from', 'detect_anomaly', 'interval',
    #                 'test_iterations', 'save_iterations', 'quiet', 'checkpoint_iterations', 'start_checkpoint']
    # for param in other_params:
    #     if hasattr(args, param):
    #         value = getattr(args, param)
    #         print(f"  {param:30} : {value}")
    #
    # print("=" * 80 + "\n")

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    if args.training_stage_from == 1:
        print("=== Starting Stage 1: Geometry Training ===")
        training(lp.extract(args), op.extract(args), pp.extract(args),
                 args.test_iterations, args.save_iterations, args.checkpoint_iterations,
                 args.stage1_checkpoint, args.debug_from)

    elif args.training_stage_from == 2:
        print("=== Starting Stage 2: BRDF Training === assuming model_path cover model from Stage 1 firstly reconstruct the source structure")
        brdf_training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint)

    # elif args.training_stage_from == 3:
    #     print("=== Starting Stage 3: Fine-tuning ===")
    #     fine_tune_training(lp.extract(args), op.extract(args), pp.extract(args),
    #                        args.test_iterations, args.save_iterations, args.checkpoint_iterations,
    #                        args.stage2_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")