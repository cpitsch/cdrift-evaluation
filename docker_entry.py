from argparse import ArgumentParser
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--evaluate', action='store_true', help='Run only the evaluation script. Assumes that the algorithm results are already present in the container.')
    parser.add_argument('--runApproaches', action='store_true', help='Run all the approaches on the Evaluation Logs (this will take a long time). (Default Behavior)')
    parser.add_argument('--runAndEvaluate', action='store_true', help='Run all the approaches on the Evaluation Logs and evaluate them (this will take a long time). This is equivalent to using both --evaluate and --runApproaches.')
    parser.add_argument('--test-run', action='store_true', help='Run each approach on only one parameter setting for testing purposes.')
    parser.add_argument('--num-cores', type=int, required=False, help="The number of cores to use for multiprocessing. Default: cpu count - 2.")
    args = parser.parse_args()


    if args.num_cores is None:
        from multiprocessing import cpu_count
        args.num_cores = cpu_count() - 2

    if args.runAndEvaluate or args.runApproaches or not any([args.evaluate, args.runApproaches, args.runAndEvaluate]):
        # Run the approaches
        import testAll_reproducibility as testAll
        testAll.main(test_run=args.test_run, num_cores=args.num_cores)

    if args.runAndEvaluate or args.evaluate:
        import evaluate
        evaluate.evaluate()