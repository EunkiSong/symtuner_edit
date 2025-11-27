'''KLEE specific components in SymTuner

This module contains KLEE specific components for running KLEE and evaluates the generated
testcases by KLEE. This module includes executable wrappers (for KLEE, KLEE-replay, and GCov)
and SymTuner implementation for KLEE.
'''

from copy import deepcopy
from pathlib import Path
import os
import random
import subprocess as sp

from symtuner.logger import get_logger
from symtuner.symbolic_executor import SymbolicExecutor
from symtuner.symtuner import SymTuner


# ---------------------------------------------------------
#  helpers for sniffles: pgm --help 에서 랜덤 옵션 뽑기
#  + pgm별 program arguments 구성
# ---------------------------------------------------------

_opt_cache = {}  # pgm별 help 옵션 캐시


def _random_opt_from_help(pgm: str):
    """
    주어진 프로그램(pgm)의 '--help' 출력에서 옵션 후보를 모아서
    그 중 하나를 랜덤으로 리턴한다.
    실패하거나 후보가 없으면 None 리턴.
    """
    if not pgm:
        return None

    if pgm in _opt_cache and _opt_cache[pgm]:
        return random.choice(_opt_cache[pgm])

    try:
        out = sp.check_output(
            [pgm, "--help"],
            stderr=sp.STDOUT,
            text=True,
        )
    except Exception:
        # 네이티브 바이너리가 없거나 실행 실패시
        return None

    candidates = set()

    # 공백 기준으로 자르고, '-'로 시작하는 토큰들만 후보로 사용
    # (뒤에 붙은 ',', '.', ';', ')' 등은 제거)
    for tok in out.split():
        if not tok.startswith("-"):
            continue
        tok = tok.rstrip(".,;:)")
        # 도와주는 옵션들은 제외
        if tok in ("-h", "--help", "--version"):
            continue
        # 너무 이상한 것들은 제외 (예: 길이 1짜리 '-' 등)
        if tok == "-" or tok == "--":
            continue
        candidates.add(tok)

    _opt_cache[pgm] = sorted(candidates)
    if not _opt_cache[pgm]:
        return None
    return random.choice(_opt_cache[pgm])


def _build_sniffles_prog_args(pgm: str, base_prog_args, regex_token: str):
    """
    sniffles 모드에서 사용할 program arguments (bitcode 뒤에 붙는 부분)를
    pgm별 스펙에 맞게 구성한다.

    base_prog_args : 원래 SymTuner가 쓰던 program args
                     (sym-arg, sym-files, sym-stdin, sym-stdout 등)
    regex_token    : SymTuner main에서 만들어준, 이미 적절히 escape된 정규식 인자 하나
                     (예: "\"...escaped...\"")
    """
    pgm = pgm or ""
    prog_args = []

    # 편의용: 랜덤 옵션 prefix (리스트 버전)
    def _opt_list():
        opt = _random_opt_from_help(pgm)
        return [opt] if opt else []

    if pgm == "diff":
        # diff:
        #   -opt regex /home/.../diff.c /home/.../diff3.c base_prog_args
        prog_args.extend(_opt_list())
        (
            prog_args.append(regex_token)
        ) if regex_token is not None else None
        prog_args.extend([
            "/home/eunki/FeatMaker/benchmarks/diffutils-3.7/src/diff.c",
            "/home/eunki/FeatMaker/benchmarks/diffutils-3.7/src/diff3.c",
        ])
        prog_args.extend(base_prog_args)

    elif pgm == "find":
        # find:
        #   -opt regex base_prog_args
        prog_args.extend(_opt_list())
        (
            prog_args.append(regex_token)
        ) if regex_token is not None else None
        prog_args.extend(base_prog_args)

    elif pgm == "gawk":
        # gawk:
        #   base_prog_args regex /home/.../gawkmisc.c /home/.../*.c
        # (명세상 -opt 없음)
        prog_args.extend(base_prog_args)
        (
            prog_args.append(regex_token)
        ) if regex_token is not None else None
        prog_args.extend([
            "/home/eunki/FeatMaker/benchmarks/gawk-5.1.0/gawkmisc.c",
            "/home/eunki/FeatMaker/benchmarks/gawk-5.1.0/*.c",
        ])

    elif pgm == "gcal":
        # gcal:
        #   -opt /home/.../de-sdata.rc -opt regex base_prog_args
        # 여기서 -opt 들은 각각 랜덤 옵션으로 대체
        prog_args.extend(_opt_list())
        prog_args.append("/home/eunki/FeatMaker/benchmarks/gcal-4.1/data/de-sdata.rc")
        prog_args.extend(_opt_list())
        (
            prog_args.append(regex_token)
        ) if regex_token is not None else None
        prog_args.extend(base_prog_args)

    elif pgm == "sed":
        # sed:
        #   -opt regex /home/.../doc/*.texi base_prog_args
        prog_args.extend(_opt_list())
        (
            prog_args.append(regex_token)
        ) if regex_token is not None else None
        prog_args.append("/home/eunki/FeatMaker/benchmarks/sed-4.8/doc/*.texi")
        prog_args.extend(base_prog_args)

    else:
        # 기타 프로그램: 기존 동작과 동일하게 regex를 앞에 붙이기
        if regex_token is not None:
            prog_args = [regex_token] + list(base_prog_args)
        else:
            prog_args = list(base_prog_args)

    return prog_args


class GCov:
    '''GCov executable wrapper

    GCov executable wrapper class.
    '''

    def __init__(self, bin='gcov'):
        '''Create GCov executable wrapper

        Create GCov executable wrapper. This includes smoke test of GCov.

        Args:
            bin: Path to GCov.
        '''

        self.bin = bin
        self.smoke_test()
        if self.bin != 'gcov':
            get_logger().info(f'Use gcov executable at: {self.bin}')

    def smoke_test(self):
        '''Test GCov executable exists

        Test GCov executable exists.

        Raises:
            CalledProcessError: If failed to find GCov at the given `bin`.
        '''

        try:
            _ = sp.run(f'{self.bin} -version', stdout=sp.PIPE, stderr=sp.PIPE,
                       shell=True, check=True)
        except sp.CalledProcessError as e:
            get_logger().fatal(f'Failed to find gcov: {self.bin}')
            raise e
        get_logger().debug(f'gcov found: {self.bin}')

    def run(self, target, gcdas, folder_depth=1):
        '''Collect covered branches with given gcdas

        Collect the covered branch set with given gcdas.

        Args:
            target: Target binary that `gcdas` are collected from.
            gcdas: A List of `gcda` files.
            folder_depth: Depth of folders to collect gcov files. For example, if `folder_depth`
                is set to 2, gcov files are collected with the `../../**/*.gcov` pattern.

        Returns:
            A set of covered branches.
        '''

        # If no gcda, return empty set
        if len(gcdas) == 0:
            return set()

        # Move to program directory.
        original_path = Path().absolute()
        target_dir = Path(target).parent
        gcdas = [gcda.absolute() for gcda in gcdas]
        os.chdir(str(target_dir))

        # Run Gcov.
        cmd = [str(self.bin), '-b', *list(map(str, gcdas))]
        cmd = ' '.join(cmd)
        get_logger().debug(f'gcov command: {cmd}')
        _ = sp.run(cmd, stdout=sp.PIPE, stderr=sp.PIPE, shell=True, check=True)

        # Get gcov files.
        base = Path()
        for _ in range(folder_depth):
            base = base / '..'
        gcov_pattern = base / '**/*.gcov'
        gcovs = list(Path().glob(str(gcov_pattern)))
        get_logger().debug(f'found gcovs: {", ".join(map(str, gcovs))}')

        # Get covered branches.
        covered = set()
        for gcov in gcovs:
            with gcov.open(encoding='UTF-8', errors='replace') as f:
                file_name = f.readline().strip().split(':')[-1]
                for i, line in enumerate(f):
                    if ('branch' in line) and ('never' not in line) and ('taken 0%' not in line):
                        bid = f'{file_name} {i}'
                        covered.add(bid)

        # Restore original directory.
        os.chdir(str(original_path))
        return covered


class KLEE(SymbolicExecutor):
    '''KLEE executable wrapper

    KLEE executable wrapper class.
    '''

    def __init__(self, bin='klee'):
        '''Create KLEE executable wrapper

        Create KLEE executable wrapper. This includes smoke test of KLEE.

        Args:
            bin: Path to KLEE.
        '''

        self.bin = bin
        self.smoke_test()
        if self.bin != 'klee':
            get_logger().info(f'Use klee executable at: {self.bin}')

    def smoke_test(self):
        '''Test KLEE executable exists

        Test KLEE executable exists.

        Raises:
            CalledProcessError: If failed to find KLEE at the given `bin`.
        '''

        try:
            _ = sp.run(f'{self.bin} -version',  stdout=sp.PIPE, stderr=sp.PIPE,
                       shell=True, check=True)
        except sp.CalledProcessError as e:
            get_logger().fatal(f'Failed to find klee: {self.bin}')
            raise e
        get_logger().debug(f'klee found: {self.bin}')

    def run(self, target, parameters, **kwargs):
        '''Run KLEE with the given parameters

        Run KLEE and collect generated testcases (`.ktest` files).

        Args:
            target: LLVM byte code file.
            parameters: A dictionary with KLEE parameters.
            kwargs: Symbolic executor specific keyword arguments.
                - pgm (str): program name (diff, find, gawk, gcal, sed, ...)
                - sniffles_option (bool): if True, use Sniffles-style regex injection.

        Returns:
            A list of testcases (`.ktest` files) founds.

        Raises:
            CalledProcessError: If some errors occur during executing KLEE.
        '''

        target = Path(target).absolute()

        # sniffles 연동용 kwargs
        pgm = kwargs.get('pgm')
        sniffles_option = kwargs.get('sniffles_option', False)

        # Convert to absolute path if output-dir option is set
        possible_output_dir = ['-output-dir', '--output-dir']
        for output_dir_param in possible_output_dir:
            if output_dir_param in parameters.keys():
                output_dir = Path(parameters[output_dir_param]).absolute()
                parameters[output_dir_param] = str(output_dir)
                break

        # Move to program directory
        original_path = Path().absolute()
        os.chdir(str(target.parent))

        # Build command
        seed_priority_opts = []      # seed 관련 옵션
        klee_options = []
        # Program arguments: -sym-arg[s] -sym-files -sym-stdin -sym-stdout
        sym_arg_options = []
        sym_files_options = []
        sym_stdin_options = []
        sym_stdout_options = []

        space_seperate_keys = ['sym-arg', 'sym-args',
                               'sym-files', 'sym-stdin']
        sym_arg_keys = ['sym-arg', 'sym-args']
        seed_order = [
            'allow-seed-extention',
            'allow-seed-truncation',
            'seed-file',
            'seed-time',
        ]
        seed_bucket = {k: [] for k in seed_order}

        for key, values in parameters.items():
            stripped_key = key.strip('-').split()[0]
            if not isinstance(values, list):
                values = [values]
            for value in values:
                if value is None:
                    param = key
                elif stripped_key in space_seperate_keys:
                    param = f'{key} {value}'
                elif stripped_key == 'sym-stdout':
                    if value == 'off':
                        continue
                    param = key
                elif stripped_key == 'sym_regex_options':
                    # 이 키는 KLEE 옵션이 아니므로 여기서는 스킵하고,
                    # 아래에서 program arguments 구성 시 사용한다.
                    continue
                else:
                    param = f'{key}={value}'

                if stripped_key in sym_arg_keys:
                    sym_arg_options.append(param)
                elif stripped_key == 'sym-files':
                    sym_files_options.append(param)
                elif stripped_key == 'sym-stdin':
                    sym_stdin_options.append(param)
                elif stripped_key == 'sym-stdout':
                    sym_stdout_options.append(param)
                elif stripped_key in seed_bucket:
                    seed_bucket[stripped_key].append(param)
                else:
                    klee_options.append(param)

        seed_priority_opts = [p for k in seed_order for p in seed_bucket[k]]

        # --- base program args (bitcode 뒤에 붙는 인자들) ---
        base_prog_args = (
            sym_arg_options +
            sym_files_options +
            sym_stdin_options +
            sym_stdout_options
        )

        # --- sym_regex_options로부터 선택된 정규식 1개를 가져온다 (있을 때만) ---
        regex_token = None
        raw_sel = parameters.get('sym_regex_options', None)
        if isinstance(raw_sel, (list, tuple)) and len(raw_sel) == 1 and isinstance(raw_sel[0], str):
            regex_token = raw_sel[0]

        # --- sniffles 모드라면 pgm별 규칙에 따라 program arguments 구성 ---
        if sniffles_option and pgm:
            prog_args = _build_sniffles_prog_args(pgm, base_prog_args, regex_token)
        else:
            # 기존 동작: sym_regex_options가 있으면 bitcode 뒤에 그대로 붙이고,
            # 그 다음에 sym-arg / sym-files / sym-stdin / sym-stdout 순으로 붙인다.
            if regex_token is not None:
                prog_args = [regex_token] + base_prog_args
            else:
                prog_args = list(base_prog_args)

        cmd = [
            str(self.bin),
            *seed_priority_opts,
            *klee_options,
            str(target),
            *prog_args,
        ]

        # Run KLEE
        cmd = ' '.join(cmd)
        get_logger().debug(f'klee command: {cmd}')
        try:
            _ = sp.run(cmd, stdout=sp.PIPE, stderr=sp.PIPE,
                       shell=True, check=True)
        except sp.CalledProcessError as e:
            stderr = e.stderr.decode(errors='replace')
            lines = stderr.strip().splitlines()
            lastline = lines[-1] if lines else ''
            if 'KLEE' in lastline and 'kill(9)' in lastline:
                get_logger().warning(f'KLEE process kill(9)ed. Failed to terminate nicely.')
            else:
                # Log and bypass if unknown error
                possible_output_dir = ['-output-dir', '--output-dir']
                for output_dir_param in possible_output_dir:
                    if output_dir_param in parameters.keys():
                        output_dir = Path(parameters[output_dir_param])
                        break
                else:
                    output_dir = target.parent / 'klee-last'
                    if output_dir.exists():
                        output_dir = output_dir.resolve()
                    else:
                        output_dir = original_path
                log_file = output_dir / 'symtuner.log'
                get_logger().warning(f'Fail({e.returncode})ed to execute KLEE. '
                                     f'See for more details: {log_file}')
                with log_file.open('w', encoding='UTF-8') as f:
                    f.write(f'command: {cmd}\n')
                    f.write(f'return code: {e.returncode}\n')
                    f.write('\n')
                    f.write('-- stdout --\n')
                    stdout = e.stdout.decode(errors='replace')
                    f.write(f'{stdout}\n')
                    f.write('-- stderr --\n')
                    stderr = e.stderr.decode(errors='replace')
                    f.write(f'{stderr}\n')

        # Get testcases
        testcases = list(output_dir.glob('*.ktest'))
        testcases = [tc.absolute() for tc in testcases]

        # Restore original directory.
        os.chdir(str(original_path))

        return testcases
    def get_time_parameter(self):
        '''Paramter to set time budget

        Parameter to set time budget.

        Returns:
            A parameter to set time budget (`-max-time`).
        '''

        return '-max-time'


class KLEEReplay:
    '''KLEE replay executable wrapper

    KLEE replay executable wrapper class.
    '''

    def __init__(self, bin='klee-replay'):
        '''Create KLEE replay executable wrapper

        Create KLEE replay executable wrapper. This includes smoke test of KLEE replay.

        Args:
            bin: Path to KLEE replay.
        '''

        self.bin = bin
        self.smoke_test()
        if self.bin != 'klee-replay':
            get_logger().info(f'Use klee-replay executable at: {self.bin}')

    def smoke_test(self):
        '''Test KLEE replay executable exists

        Test KLEE replay executable exists.

        Raises:
            CalledProcessError: If failed to find KLEE replay at the given `bin`.
        '''

        try:
            _ = sp.run(f'which {self.bin}',  stdout=sp.PIPE, stderr=sp.PIPE,
                       shell=True, check=True)
        except sp.CalledProcessError as e:
            get_logger().fatal(f'Failed to find klee-replay: {self.bin}')
            raise e
        get_logger().debug(f'klee-replay found: {self.bin}')

    def run(self, target, testcase, error_type=None, folder_depth=1):
        '''Replay the testcase

        Replay the given KLEE testcase (`.ktest` file) with the GCov object. Then, collect the bugs
        and `gcda` files.

        Args:
            target: Target executable with GCov configuration.
            testcase: Testcase to replay
            error_type: A list of error types consider. If not specificed, 2 types of bugs will be
                considered: `CRASHED signal 11` and `CRASHED signal 6`.
            folder_depth: Depth of folders to collect gcov files. For example, if `folder_depth`
                is set to 2, gcda files are collected with the `../../**/*.gcda` pattern.

        Returns:
            A tuple of bugs, and `gcda` files. First element of the tuple is the found bugs, and
            the second element is collected `gcda` files.
        '''

        target = Path(target).absolute()
        testcase = Path(testcase).absolute()

        # Move to program directory.
        original_path = Path().absolute()
        os.chdir(str(target.parent))

        # Error types interested in
        if error_type is None:
            error_type = ['CRASHED signal 11', 'CRASHED signal 6']
        if isinstance(error_type, str):
            error_type = [error_type]

        # Run KLEE-replay
        cmd = [str(self.bin), str(target), str(testcase)]
        cmd = ' '.join(cmd)
        get_logger().debug(f'klee-replay command: {cmd}')
        process = sp.Popen(cmd, stdout=sp.PIPE, stderr=sp.PIPE, shell=True)
        errors = set()
        try:
            _, stderr = process.communicate(timeout=0.1)
            lastline = str(stderr.splitlines()[-1])

            # Find error types
            for error in error_type:
                if error in lastline:
                    errs = list(testcase.parent.glob(testcase.stem + '.*.err'))

                    for err in errs:
                        with err.open(encoding='UTF-8', errors='replace') as f:
                            lines = f.readlines()
                            file_name = lines[1].split()[1]
                            line_num = lines[2].split()[1]
                            err_type = f'{file_name} {line_num}'
                            errors.add(err_type)
        except sp.TimeoutExpired:
            get_logger().warning(f'KLEE replay timeout: {testcase}')
        finally:
            process.kill()

        # Find *.gcda files.
        base = Path()
        for _ in range(folder_depth):
            base = base / '..'
        gcda_pattern = base / '**/*.gcda'
        gcdas = list(target.parent.glob(str(gcda_pattern)))
        gcdas = [gcda.absolute() for gcda in gcdas]

        # Restore original directory.
        os.chdir(str(original_path))

        return errors, gcdas


class KLEESymTuner(SymTuner):
    '''SymTuner implementation for KLEE

    SymTuner impelmentation for KLEE. This includes, defaults parameter spaces, how to evalutate
    testcaes, and parameter space updates for `-seed-file` field.
    '''

    def __init__(self, klee_replay=None, gcov=None, k_seeds=10, *args, **kwargs):
        '''Create a new SymTuner for KLEE

        Args:
            klee_replay: A reference to klee-replay executable. Must be a string or
                a `symtuner.klee.KLEEReplay` instance.
            gcov: A reference to gcov executable. Must be a string or
                a `symtuner.klee.GCov` instance.
            args: Any positional arguments that are needed to initialize
                `symtuner.symtuner.SymTuner` object.
            kwargs: Any keyword arguments that are needed to initialize
                `symtuner.symtuner.SymTuner` object.
        '''

        super(KLEESymTuner, self).__init__(*args, **kwargs)

        if klee_replay is None:
            klee_replay = KLEEReplay()
        elif isinstance(klee_replay, str):
            klee_replay = KLEEReplay(klee_replay)
        self.klee_replay = klee_replay
        if gcov is None:
            gcov = GCov()
        elif isinstance(gcov, str):
            gcov = GCov(gcov)
        self.gcov = gcov

        self.k_seeds = k_seeds

    def sample(self, policy=None):
        '''Sample a set of parameters to use

        Sampling with 2 policies: exploit and eplore.

        Args:
            policy: Sampling policy. One of 'exploit' and 'explore'. If not set, sampling with
            [exploit_portion, 1 - exploit_portion] internally.

        Returns:
            A dictionary of sampled parameters.
        '''

        parameters = super(KLEESymTuner, self).sample(policy)

        # If -seed-file option defined check special options
        if '-seed-file' in parameters.keys() or '--seed-file' in parameters.keys():
            key = '-seed-file' if '-seed-file' in parameters.keys() else '--seed-file'
            value = parameters[key]

            if value == 'random_from_all':
                testcases = [tc for _, _, tc, _ in self.data]
                if len(testcases) > 0:
                    testcase = random.choice(testcases)
                    parameters[key] = str(testcase)
                else:
                    del parameters[key]

        return parameters

    def add(self, target, parameters, testcases, evaluation_kwargs=None):
        '''Evaluate and update data

        Update data and space and variables for -seed-file.

        Args:
            target: A target program to evaluate with.
            paramters: A set of parameters used to generated testcases.
            testcases: Testcases genereted with parameters.
            evaluation_kwargs: A dictionary of keyword arguments pass to evaluate method.

        Returns:
            Self object for chaining. All updates is recorded in the object.
        '''

        super(KLEESymTuner, self).add(target, parameters, testcases,
                                      evaluation_kwargs)

        # Skip if -seed-file is not defined in space
        if '-seed-file' not in self.space.keys() and '--seed-file' not in self.space.keys():
            return self

        # Find buggy testcases
        buggy_seeds = []
        found_bugs = []
        for _, bugs, tc, _ in self.data[::-1]:
            for bug in bugs:
                if bug not in found_bugs:
                    found_bugs.append(bug)
                    buggy_seeds.append(tc)

        # Find top k testcases that covers most
        accumulated_coverage = set()
        copied_data = deepcopy(self.data)
        top_k_seeds = []

        for _ in range(self.k_seeds):
            if len(copied_data) == 0:
                break
            copied_data = sorted(copied_data,
                                 key=lambda elem: len(elem[0]),
                                 reverse=True)
            top_cov, _, tc, _ = copied_data.pop(0)
            if len(top_cov) > 0:
                accumulated_coverage = accumulated_coverage | top_cov
                copied_data = [(cov - accumulated_coverage, bug, tc, param)
                               for cov, bug, tc, param in copied_data]
                top_k_seeds.append(tc)
            else:
                break

        # Update space for -seed-file
        key = '-seed-file' if '-seed-file' in self.space.keys() else '--seed-file'
        seed_files = buggy_seeds + top_k_seeds
        self.space[key] = (seed_files, self.space[key][1])
        for seed in seed_files:
            if seed not in self.cnts[key].keys():
                self.cnts[key][seed] = 0
        return self

    def evaluate(self, target, testcase, folder_depth=1):
        '''Evaluate the given testcase

        Evalutate the given testcase with KLEE replay and GCov.

        target: A target program to evaluate with. Must be compiled with GCov settings.
        testcase: A testcase (`.ktest` file) to replay.
        folder_depth: Depth of folders to collect gcov files. This argument is passed
            to `GCov.run` and `KLEEReplay.run` methods.

        Returns:
            A tuple of covered braches and found bugs. The first element of the tuple
            is a set of covered branches and the second element is a set of found bugs.
        '''

        # Remove existing gcdas and gcovs
        base = Path(target).parent
        for _ in range(folder_depth):
            base = base / '..'
        cmd = ['rm', '-f', str(base / '**/*.gcda'), str(base / '**/*.gcov')]
        cmd = ' '.join(cmd)
        get_logger().debug(f'gcda gcov clean up command: {cmd}')
        _ = sp.run(cmd, shell=True, check=True)
        errors, gcdas = self.klee_replay.run(target, testcase,
                                             folder_depth=folder_depth)
        branches = self.gcov.run(target, gcdas, folder_depth=folder_depth)
        return branches, errors

    @classmethod
    def get_default_space(cls):
        '''Default tuning space for KLEE

        Default tuning space for KLEE. This will used if user does not specify spaces.

        Returns:
            A dictionary that contains default tuning space.
        '''

        search_heuristics = ['nurs:cpicnt', 'nurs:qc', 'nurs:covnew', 'random-path', 'bfs',
                             'nurs:md2u', 'nurs:icnt', 'nurs:depth', 'random-state', 'dfs']
        space = {
            # boolean
            '-simplify-sym-indices': (['true', 'false'], 1),
            '-use-forked-solver': (['true', 'false'], 1),
            '-use-cex-cache': (['true', 'false'], 1),
            '-max-memory-inhibit': (['true', 'false'], 1),
            '-optimize': (['true', 'false'], 1),
            '-sym-stdout': (['on', 'off'], 1),

            # integer
            '-max-memory': ([500, 1000, 1500, 2000, 2500], 1),
            '-max-sym-array-size': ([3000, 3500, 4000, 4500, 5000], 1),
            '-max-instruction-time': ([10, 20, 30, 40, 50], 1),
            '-max-static-fork-pct': ([0.25, 0.5, 1, 2, 4], 1),
            '-max-static-solve-pct': ([0.25, 0.5, 1, 2, 4], 1),
            '-max-static-cpfork-pct': ([0.25, 0.5, 1, 2, 4], 1),
            '-batch-instructions': ([6000, 8000, 10000, 12000, 14000], 1),
            '-sym-arg': ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 5),
            '-sym-files 1': ([4, 8, 12, 16, 20], 1),
            '-sym-stdin': ([4, 8, 12, 16, 20], 1),

            # string
            '-seed-file': ([], 1),
            '-search': (search_heuristics, 1),
            '-switch-type': (['simple', 'internal'], 1),
            '-external-calls': (['concrete', 'all'], 1),
        }
        return space

    @classmethod
    def get_default_default_parameters(cls):
        '''Default default arguments for KLEE

        Default default arguments for KLEE. This will used if user does not specify defaults arguments.

        Returns:
            A dictionray that containes default default arguments.
        '''

        defaults = {
            '-output-module': 'false',
            '-output-source': 'false',
            '-output-stats': 'false',
            '-use-batching-search': None,
            '-posix-runtime': None,
            '-only-output-states-covering-new': None,
            '-watchdog': None,
            '-allow-seed-extension': None,
            '-allow-seed-truncation': None,
            '-ignore-solver-failures': None,
            '-libc': 'uclibc',
            '-disable-inlining': None,
        }
        return defaults
