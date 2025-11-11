import time
import pyEDM as pe
import sys
from utils.arg_parser import get_parser
from utils.config_parser import load_config

from data_obj.data_objects import *
from data_obj.data_var import *
from utils import process_output as po


def decide_file_handling(args, file_exists: bool, modify_datetime=None) -> tuple[bool, bool]:
    """
    Decide (run_continue, overwrite) based on:
      - args.override       (bool)
      - args.write          ("append" or "replace")
      - args.datetime_flag  (optional datetime cutoff)
      - file_exists         (bool)
      - modify_datetime     (file’s mtime as datetime or None)
    """
    # default to running and overwriting
    run_continue = True
    overwrite    = True

    # 1) if the file exists & no override → maybe skip
    if file_exists and not args.override:
        if args.datetime_flag is not None:
            try:
                if modify_datetime >= args.datetime_flag:
                    # file is fresh/newer than cutoff → skip
                    run_continue = False
                    overwrite    = False
                    return run_continue, overwrite
            except Exception:
                # if compare fails, ignore and proceed
                pass
        else:
            # no datetime_flag → skip unconditionally
            run_continue = False
            overwrite    = False
            return run_continue, overwrite

    # 2) if file exists & user asked to append → run & append
    if file_exists and args.write == "append":
        run_continue = True
        overwrite    = False
        print("Appending to existing file.", file=sys.stdout, flush=True)
        return run_continue, overwrite

    # 3) otherwise → run & overwrite
    return run_continue, overwrite

def print_log_line(script, function, log_line, log_type='info'):
    if log_type == 'error':
        file_pointer = sys.stderr
    else:
        file_pointer = sys.stdout
    timestamp = datetime.datetime.now()
    print(timestamp.strftime('%Y-%m-%d %H:%M:%S'), log_line, f'{script}: {function}', file=file_pointer, flush=True)
    return time.time()

def run_experiment(arg_tuple):
    '''
    Run EDM CCM experiment based on CCMConfig object
    Parameters:
        arg_tuple: (ccm_obj, cpu_count, self_predict, time_offset)
            ccm_obj: CCMConfig object
            cpu_count: number of CPUs to use
            self_predict: boolean, whether to use self prediction
            time_offset: integer, offset to add to run_id
    Returns:
        (ccm_out_df, df_path)


    '''

    ccm_obj, cpu_count, self_predict, time_offset = arg_tuple

    time_var = ccm_obj.time_var
    run_id = int(time.time() * 1000) + time_offset

    df_path = ccm_obj.file_path#output_dir / df_csv_name
    start_time = print_log_line('run_edm_carc_pool2_config_toObj', 'run_experiment',
                                     f'{df_path} exists: {df_path.exists()}, starting start_ind {start_ind}, pset_id {ccm_obj.pset_id}, col_var_id {ccm_obj.col_var_id}, target_var_id {ccm_obj.target_var_id}, E {ccm_obj.E}, tau {ccm_obj.tau}, lag {ccm_obj.lag}, knn {ccm_obj.knn}, Tp {ccm_obj.Tp}, sample {ccm_obj.sample}, weighted {ccm_obj.weighted}, train_ind_i {ccm_obj.train_ind_i}, surr_var {ccm_obj.surr_var}, surr_num {ccm_obj.surr_num}', 'info')

    try:
        pred_num = config.ccm_config.pred_num
    except:
        pred_num = None

    # note: at some point "embedded=False" will not always be correct
    # cpu_count = 1 for HPC runs where resources are allocated less flexibly
    ccm_out = pe.CCM(dataFrame=ccm_obj.df,
                     E=ccm_obj.E, Tp=ccm_obj.Tp, tau=-ccm_obj.tau,
                     exclusionRadius=ccm_obj.exclusion_radius,
                     knn=ccm_obj.knn, verbose=False,
                     columns=ccm_obj.col_var,
                     target=ccm_obj.target_var,
                     libSizes=ccm_obj.libsizes,
                     sample=ccm_obj.sample,
                     embedded=False, seed=None,
                     weighted=ccm_obj.weighted, includeData=True, returnObject=True,
                     pred_num=pred_num,
                     num_threads=cpu_count,
                     showPlot=False, noTime=False, selfPredict=self_predict)

    ccm_out_df = pd.concat(
        [po.unpack_ccm_output(ccm_out.CrossMapList[ip]) for ip in range(len(ccm_out.CrossMapList))])
    ccm_out_df = po.add_meta_data(ccm_out, ccm_out_df, ccm_obj.train_ind_i, ccm_obj.train_ind_f, lag=ccm_obj.lag)
    ccm_out_df['lag'] = ccm_obj.lag

    ccm_obj.output_path.mkdir(parents=True, exist_ok=True)

    ccm_out_df['run_id'] = run_id
    ccm_out_df['pset_id'] = ccm_obj.pset_id

    print('!\tfinish', 'start index:', start_ind, ccm_obj.pset_id, f'time elapsed: {time.time() - start_time}',
          ccm_obj.col_var_id, ccm_obj.target_var_id,
          'E, tau, lag= ', ccm_obj.E, ccm_obj.tau, ccm_obj.lag,
          file=sys.stdout, flush=True)

    # return [logs]
    return ccm_out_df, df_path


def write_to_file(ccm_out_df, df_path, overwrite=False):
    remove_cols = ['Tp_lag_total', 'sample', 'weighted', 'train_ind_0', 'run_id', 'ind_f', 'Tp_flag', 'train_len',
                   'train_ind_i']
    ccm_out_df = ccm_out_df[[col for col in ccm_out_df.columns if col not in remove_cols]].copy()
    df_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        ccm_out_df_0 = pd.read_csv(df_path, index_col=0)
        if overwrite == False:
            ccm_out_df_0 = ccm_out_df_0[[col for col in ccm_out_df_0.columns if col not in remove_cols]].copy()
            ccm_out_df = pd.concat([ccm_out_df_0, ccm_out_df])
            ccm_out_df.reset_index(drop=True, inplace=True)
    except:
        pass

    ccm_out_df.to_csv(df_path)
    if os.path.exists(df_path):
        print('!\twrote to file: ', df_path)
    else:
        print('x\tfailed to write to file: ', df_path, file=sys.stderr, flush=True)
        print('x\tfailed to write to file: ', df_path, file=sys.stdout, flush=True)



if __name__ == '__main__':
    # grab parameter file
    parser = get_parser()
    args = parser.parse_args()

    if args.project is not None:
        proj_name = args.project
    else:
        print('project name is required', file=sys.stdout, flush=True)
        print('project name is required', file=sys.stderr, flush=True)
        sys.exit(0)

    proj_dir = Path(os.getcwd()) / proj_name
    gen_config = 'proj_config'
    config = load_config(proj_dir / f'{gen_config}.yaml')

    if args.parameters is not None:
        parameter_flag = args.parameters
        parameter_dir = proj_dir / 'parameters'
        parameter_path = parameter_dir / f'{parameter_flag}.csv'
        parameter_df = pd.read_csv(parameter_path)
    else:
        print('parameters are required', file=sys.stdout, flush=True)
        print('parameters are required', file=sys.stderr, flush=True)
        sys.exit(0)

    flags = []
    if args.flags is not None:
        flags += args.flags

    if args.cpus is not None:
        cpu_count = int(args.cpus)
    else:
        cpu_count = 4

    #
    num_inds = 10
    if args.inds is not None:
        start_ind = int(args.inds[-1])
        if len(args.inds) > 1:
            num_inds = int(args.inds[-2])
    end_ind = start_ind + num_inds

    if start_ind not in parameter_df.index:  #len(parameter_ds):
        print('start_ind is not in available indexes', file=sys.stdout, flush=True)
        sys.exit(0)

    spec_config = {}
    if args.config is not None:
        spec_config_yml = args.config
        config = load_config(proj_dir / f'{spec_config_yml}.yaml')

    parameter_df = parameter_df.loc[start_ind:end_ind, :].copy()
    for int_col in ['E', 'tau', 'lag', 'knn', 'train_ind_i', 'Tp', 'Tp_lag_total', 'sample', 'id']:
        if int_col in parameter_df.columns:
            parameter_df[int_col] = parameter_df[int_col].astype(int)
    parameter_ds = parameter_df.to_dict(orient='records')

    second_suffix = ''
    if args.test:
        second_suffix = f'_{int(time.time() * 1000)}'

    calc_location = set_calc_path(args, proj_dir, config, second_suffix)
    output_dir = set_output_path(args, calc_location, config)
    calc_location.mkdir(parents=True, exist_ok=True)

    logs = []
    log_element = []
    self_predict = False

    loc = check_location(local_word='jlanders')
    # if Path('/Users/jlanders').exists():
    if loc == 'local':
        print('local', file=sys.stdout, flush=True)

        arg_tuples = []
        for time_offset, pset_d in enumerate(parameter_ds):
            print(pset_d, file=sys.stdout, flush=True)
            pset_d= {k: v[0] if isinstance(v, list) and len(v) == 1 else v for k, v in pset_d.items()}
            if 'pset_id' not in pset_d:
                if 'id' in pset_d:
                    pset_d['pset_id'] = pset_d['id']

            new_rc = RunConfig(pset_d)
            ccm_obj = CCMConfig(new_rc, config, proj_dir=proj_dir)
            pset_exists, stem_exists = ccm_obj.check_run_exists()
            # this is strong existence criteria... if want to check for stem existence, use stem_exists

            run_continue, overwrite = decide_file_handling(args, pset_exists)

            if run_continue == False:
                print(ccm_obj.file_name, '\n\tskipping: ', ccm_obj.pset_id, ccm_obj.col_var_id, ccm_obj.target_var_id, f'E={ccm_obj.E}, tau={ccm_obj.tau}, lag={ccm_obj.lag}', file=sys.stdout, flush=True)
                continue

            print('how many cores:', cpu_count, file=sys.stdout, flush=True)
            candidate_tuple = (ccm_obj, cpu_count, self_predict, start_ind + time_offset)#(ccm_obj, time_offset, start_ind + time_offset, config, cpu_count, self_predict)
            write_to_file(*run_experiment(candidate_tuple), overwrite=overwrite)

    else:
        parameter_ds = parameter_ds[0]
        time_offset = 1

        pset_d = parameter_ds

        new_rc = RunConfig(pset_d)
        ccm_obj = CCMConfig(new_rc, config, proj_dir=proj_dir)
        pset_exists, stem_exists = ccm_obj.check_run_exists()
        # this is strong existence criteria... if want to check for stem existence, use stem_exists

        run_continue, overwrite = decide_file_handling(args, pset_exists)

        if run_continue == False:
            print(ccm_obj.file_name, '\n\tskipping: ', ccm_obj.pset_id, ccm_obj.col_var_id, ccm_obj.target_var_id,
                  f'E={ccm_obj.E}, tau={ccm_obj.tau}, lag={ccm_obj.lag}', file=sys.stdout, flush=True)
        else:
            print('how many cores:', cpu_count, file=sys.stdout, flush=True)
            candidate_tuple = (ccm_obj, cpu_count, self_predict,
                               start_ind + time_offset)  # (ccm_obj, time_offset, start_ind + time_offset, config, cpu_count, self_predict)
            write_to_file(*run_experiment(candidate_tuple), overwrite=overwrite)
