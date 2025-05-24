import argparse

from pm_kvq.evaluation.judge import judge_aime, judge_cmimc, judge_livecodebench

parser = argparse.ArgumentParser()
parser.add_argument("--benchmark", type=str, help="Benchmark name", default="aime", choices=["aime", "cmimc", "livecodebench"])
parser.add_argument("--version", type=str, help="Benchmark version", default="2024")
parser.add_argument("--responses_dir", help="Path to the model responses", type=str)
args = parser.parse_args()

if args.benchmark == "aime":
    results = judge_aime(args.responses_dir, args.version)
    print(f"pass@1: {results[0]:.2f}")
    print(f"Voting acc: {results[1]:.2f}")
elif args.benchmark == "cmimc":
    results = judge_cmimc(args.responses_dir, args.version)
    print(f"pass@1: {results[0]:.2f}")
    print(f"Voting acc: {results[1]:.2f}")
elif args.benchmark == "livecodebench":
    results = judge_livecodebench(args.responses_dir, args.version)
    print(f"pass@1: {results['pass@1']:.2f}")
