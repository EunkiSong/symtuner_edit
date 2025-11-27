'''Module that defines console script entries

This module defines console script entries. This module includes console script entry
for SymTuner for KLEE.
'''

from pathlib import Path
import argparse
import json
import shutil
import sys
import os
import glob
import re

from symtuner.klee import KLEE
from symtuner.klee import KLEESymTuner
from symtuner.logger import get_logger
from symtuner.symtuner import TimeBudgetHandler


#############################
#  Sniffles 전용 유틸
#############################

def _ensure_regex_file(top_dir: str, mode: str) -> str:
    """
    rand_regex.txt (50 lines) 생성 또는 재사용.
    저장 위치: <top_dir>/regex_gen/rand_regex.txt
    """
    out_dir = Path(top_dir) / "regex_gen"
    os.makedirs(out_dir, exist_ok=True)
    out_file = out_dir / "rand_regex.txt"

    # 없거나 비어 있으면 생성
    if not (out_file.exists() and out_file.stat().st_size > 0):
        os.system(f'regexgen --mode {mode} --output "{out_file}" -c 50')

    return str(out_file)


def _read_lines(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return [ln.strip() for ln in f if ln.strip()]
    except Exception:
        return []


def _escape_inner_quotes(s: str) -> str:
    """
    정규표현식 내부의 ', " 그리고 ` 앞에 백슬래시를 붙여 준다.
    이미 \', \" 혹은 \` 인 경우는 다시 이스케이프하지 않는다.
    (즉, 바로 앞 문자가 '\'가 아니면 '\'를 붙인다.)
    """
    result = []
    prev_backslash = False
    for ch in s:
        if ch in ("'", '"', '`'):
            if not prev_backslash:
                result.append('\\')
        result.append(ch)
        prev_backslash = (ch == '\\')
    return ''.join(result)


#############################
#  메인
#############################

def main(argv=None):

    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser()

    # Executable settings
    executable = parser.add_argument_group('executable settings')
    executable.add_argument('--klee', default='/home/eunki/FeatMaker/klee/build/bin/klee',
                            help='Path to "klee" executable (default=klee)')
    executable.add_argument('--klee-replay', default='/home/eunki/FeatMaker/klee/build/bin/klee-replay',
                            help='Path to "klee-replay" executable (default=klee-replay)')
    executable.add_argument('--gcov', default='gcov',
                            help='Path to "gcov" executable (default=gcov)')

    # Hyperparameters
    hyper = parser.add_argument_group('hyperparameters')
    hyper.add_argument('-s', '--search-space', default=None,
                       help='Json file defining parameter search space')
    hyper.add_argument('--exploit-portion', default=0.7, type=float,
                       help='Portion of exploitation in SymTuner (default=0.7)')
    hyper.add_argument('--step', default=20, type=int,
                       help='The number of runs before increasing small budget (default=20)')
    hyper.add_argument('--minimum-time-portion', default=0.005, type=float,
                       help='Minimum portion for one iteration (default=0.005)')
    hyper.add_argument('--increase-ratio', default=2, type=float,
                       help='Factor to increase small budget (default=2)')
    hyper.add_argument('--minimum-time-budget', default=30, type=int,
                       help='Minimum time budget per run (default=30)')
    hyper.add_argument('--exploration-steps', default=20, type=int,
                       help='Iterations focusing only on exploration (default=20)')

    hyper.add_argument(
        '--seed-mode',
        nargs='?',
        const=True,
        choices=['top600', 'bot600', 'top600_switch_symtuner'],
        default=False,
        help='Seed selection mode for SymTuner'
    )

    # Others
    parser.add_argument('-d', '--output-dir', default='symtuner-out',
                        help='Directory to store the generated files (default=symtuner-out)')
    parser.add_argument('--generate-search-space-json', action='store_true',
                        help='Generate the json file defining parameter spaces used in our ICSE\'22 paper')
    parser.add_argument('--debug', action='store_true',
                        help='Log debug messages')
    parser.add_argument('--gcov-depth', default=1, type=int,
                        help='Depth to search for gcda/gcov files from gcov_obj (default=1)')
    parser.add_argument('--max_testcase', default=None, type=int,
                        help='Maximum number of .ktest files to stop execution')
    parser.add_argument('--pgm', default=None,
                        help='Program name (e.g., find, grep, gcal, sed, gawk)')
    parser.add_argument('--gcov_num', default=None, type=int,
                        help='Coverage build number to replace <gcov_num> in config.json')

    # sniffles: 랜덤 regex를 직접 인자로 붙이는 모드
    parser.add_argument('--sniffles-option', action='store_true',
                        help='Enable Sniffles mode (inject random regex as positional argument).')

    # Required
    required = parser.add_argument_group('required')
    required.add_argument('-t', '--budget', type=int,
                          help='Total time budget in seconds')
    required.add_argument('llvm_bc', nargs='?',
                          help='LLVM bitcode file for KLEE')
    required.add_argument('gcov_obj', nargs='?',
                          help='Executable with gcov support')

    args = parser.parse_args(argv)

    # ------------------------------
    #  config.json 로딩 (--pgm)
    # ------------------------------
    llvm_bc = args.llvm_bc
    gcov_obj = args.gcov_obj

    if args.pgm:
        with open('./config.json', 'r') as f:
            config = json.load(f)

        if args.pgm not in config:
            print(f"[Error] '{args.pgm}' missing in config.json")
            sys.exit(1)

        pg = config[args.pgm]

        # gcov_obj
        if '<gcov_num>' in pg["gcov_obj"]:
            if args.gcov_num is None:
                print("[Error] --gcov_num required because <gcov_num> is in config.json")
                sys.exit(1)
            gcov_obj = pg["gcov_obj"].replace("<gcov_num>", str(args.gcov_num))
        else:
            gcov_obj = pg["gcov_obj"]

        llvm_bc = pg["llvm_bc"]

    if llvm_bc is None or gcov_obj is None or args.budget is None:
        print("Error: missing required args -t, llvm_bc, gcov_obj")
        sys.exit(1)

    if args.debug:
        get_logger().setLevel('DEBUG')

    # ------------------------------
    #  출력 디렉토리 준비
    # ------------------------------
    outdir = Path(args.output_dir)
    if outdir.exists():
        shutil.rmtree(outdir)
        get_logger().warning(f'Existing output directory deleted: {outdir}')
    outdir.mkdir(parents=True)

    coverage_csv = outdir / "coverage.csv"
    coverage_csv.touch()
    get_logger().info(f'Coverage will be recorded at "{coverage_csv}"')

    found_bugs_txt = outdir / "found_bugs.txt"
    found_bugs_txt.touch()
    get_logger().info(f'Found bugs will be recorded at "{found_bugs_txt}"')

    # ------------------------------
    #  seed-mode 처리
    # ------------------------------
    initial_seed_files = []
    if args.seed_mode:
        if not args.pgm:
            parser.error("--seed-mode requires --pgm to be specified")

        seed_dir = os.path.join(
            "/home/eunki/Comparing_testcases/RQ0/seeds",
            args.seed_mode,
            args.pgm
        )
        if not os.path.isdir(seed_dir):
            print(f"Warning: seed directory '{seed_dir}' not found; initial_seed_files set to empty list")
            initial_seed_files = []
        else:
            candidates = glob.glob(os.path.join(seed_dir, f"{args.pgm}_*.ktest"))
            pairs = []
            for p in candidates:
                m = re.search(r'_(\d+)\.ktest$', os.path.basename(p))
                if m:
                    pairs.append((int(m.group(1)), p))
            initial_seed_files = [p for _, p in sorted(pairs)]
            skipped = set(candidates) - set(initial_seed_files)
            if skipped:
                print(f"Note: skipped {len(skipped)} non-numeric {args.pgm}_*.ktest files")

        print(f"Loaded {len(initial_seed_files)} initial seed files from {seed_dir}")

    # ------------------------------
    #  Sniffles: rand_regex.txt 준비 (+ 내부 escape)
    # ------------------------------
    sniffles_regexes = None
    if args.sniffles_option:
        if not args.pgm:
            print("[Error] --sniffles-option requires --pgm to be specified")
            sys.exit(1)

        mode = "bre" if args.pgm == "diff" else "ere"
        regex_file = _ensure_regex_file(str(outdir), mode)

        raw_regexes = _read_lines(regex_file)
        if not raw_regexes:
            print("[Error] rand_regex.txt empty")
            sys.exit(1)

        # 내부 ', ", ` 를 백슬래시로 이스케이프한 뒤, 외부를 "..." 로 감싼 문자열로 저장
        # → 이 문자열 하나가 셸 기준 "한 개의 인자"로 들어가도록 보장
        sniffles_regexes = [f"\"{_escape_inner_quotes(r)}\"" for r in raw_regexes]

    # ------------------------------
    #  초기화
    # ------------------------------
    symbolic_executor = KLEE(args.klee)
    symtuner = KLEESymTuner(
        args.klee_replay,
        args.gcov,
        10,
        args.search_space,
        args.exploit_portion
    )
    evaluation_argument = {'folder_depth': args.gcov_depth}

    time_budget = TimeBudgetHandler(
        args.budget,
        args.minimum_time_portion,
        args.step,
        args.increase_ratio,
        args.minimum_time_budget
    )

    get_logger().info('All configuration loaded. Start testing.')

    # ------------------------------
    #  메인 루프
    # ------------------------------
    for i, tb in enumerate(time_budget):

        iteration_dir = outdir / f"iteration-{i}"

        # search-space 샘플링 (튜닝 파라미터만)
        policy = 'explore' if i < args.exploration_steps else None
        parameters = symtuner.sample(policy=policy)

        # ---- sniffles: 짝수 번째 iteration(2,4,6,...)에만 정규식 삽입 ----
        if args.sniffles_option and sniffles_regexes:
            # 사람 기준 짝수 번째 iteration -> (i+1) % 2 == 0
            if i  % 2 == 0:
                # 짝수 번째 iteration마다 regex 리스트를 순서대로 사용
                # 2번째 iteration에서 0번째 regex, 4번째에서 1번째 ... 쓰게 하려면 i // 2 사용
                idx = (i // 2) % len(sniffles_regexes)
                regex = sniffles_regexes[idx]
                # sniffles에서는 sym_regex_options에 "이미 이스케이프된 하나의 인자"를 넣고,
                # 실제 pgm별 symbolic args + -opt 랜덤 옵션 구성은 klee.py에서 처리
                parameters['sym_regex_options'] = [regex]
            else:
                # 홀수 번째 iteration에서는 regex를 사용하지 않음
                parameters.pop('sym_regex_options', None)
        else:
            # sniffles off → sym_regex_options는 사용하지 않음
            parameters.pop('sym_regex_options', None)

        # 시간/출력 디렉토리 설정
        parameters[symbolic_executor.get_time_parameter()] = tb
        parameters['-output-dir'] = str(iteration_dir)

        # seed-mode 사용 시 seed 파일 지정
        if args.seed_mode and initial_seed_files:
            if i < len(initial_seed_files):
                parameters['--seed-file'] = str(initial_seed_files[i])
            else:
                print(f"[Warning] Not enough seed files for iteration {i}; skipping --seed-file.")
            parameters['--seed-time'] = "30s"

        # KLEE 실행
        # → klee.py에서 sniffles_option, pgm, sym_regex_options를 조합해서
        #    pgm별 symbolic args와 -opt 랜덤 옵션을 구성하도록 패치하면 됨
        testcases = symbolic_executor.run(
            llvm_bc,
            parameters,
            pgm=args.pgm,
            sniffles_option=args.sniffles_option
        )

        # 결과 수집:
        #  - sniffles로 들어간 임의 regex는 search-space에 정의되어 있지 않으므로
        #    count_used_parameters에서 KeyError가 나지 않게 통계용 딕셔너리에서는 제거
        if args.sniffles_option:
            params_for_stats = dict(parameters)
            params_for_stats.pop('sym_regex_options', None)
            symtuner.add(gcov_obj, params_for_stats, testcases, evaluation_argument)
        else:
            symtuner.add(gcov_obj, parameters, testcases, evaluation_argument)

        elapsed = time_budget.elapsed
        coverage, bugs = symtuner.get_coverage_and_bugs()
        get_logger().info(
            f'Iteration: {i + 1} '
            f'Time budget: {tb} '
            f'Time elapsed: {elapsed} '
            f'Coverage: {len(coverage)} '
            f'Bugs: {len(bugs)}'
        )
        with coverage_csv.open('a') as stream:
            stream.write(f'{elapsed}, {len(coverage)}\n')
        with found_bugs_txt.open('w') as stream:
            stream.writelines(
                (f'Testcase: {Path(symtuner.get_testcase_causing_bug(bug)).absolute()} '
                 f'Bug: {bug}\n' for bug in bugs)
            )

    coverage, bugs = symtuner.get_coverage_and_bugs()
    get_logger().info(f'SymTuner done. Achieve {len(coverage)} coverage and found {len(bugs)} bugs.')
