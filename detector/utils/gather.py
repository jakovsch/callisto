import os, io, shlex, pandas as pd
from pandas import Timestamp, Timedelta, Interval
from functools import reduce
from subprocess import run
from glob import glob
from re import escape

DATAURL = 'http://soleil80.cs.technik.fhnw.ch/solarradio/data'
DATADIR = os.path.join(os.getcwd(), 'data')
LISTDIR = os.path.join('BurstLists', '2010-yyyy_Monstein')
FITSDIR = os.path.join('2002-20yy_Callisto')
wget_default = f'wget -nv -nc -np -nH -e robots=off -r --cut-dirs=2'
sel_default = {
    'typem': lambda s: (s.removesuffix('?') if len(s) > 1 else s).split('/')[0],
    'typef': lambda s: s not in ('RBR', 'CAU', 'CTMCAU') and '?' not in s,
    'statm': lambda s: s,
    'statf': lambda s: s not in ('*', 'e-Callisto') and all(i.isalnum() for i in s.replace('_', '-').split('-')),
}

def _digit_range(a, b):
    dig = [divmod(n, 10) for n in (a, b)]
    wrap = dig[1][0] > dig[0][0]
    return (
        (dig[0][0], dig[0][1], 9),
        *((i, 0, 9) for i in range(dig[0][0]+1, dig[1][0])),
        (dig[1][0], 0, dig[1][1]),
    ) if wrap else (
        (dig[0][0], dig[0][1], dig[1][1]),
    )

def get_lists(path, filt=r'202[0-9]/.*'):
    args = f'--accept txt --accept-regex "{filt}" {DATAURL}/BurstLists/2010-yyyy_Monstein'
    prefix = f' -P {path!r} '
    run(shlex.split(wget_default+prefix+args), check=True)

def get_fits(path, blist):
    toregex = lambda r: f'{r[0]}[{r[1]}-{r[2]}]' if r[1] != r[2] else f'{r[0]}{r[1]}'
    for i in blist.index:
        ev = blist.loc[i]
        date = ev['date']
        start, end = ev['time'].left, ev['time'].right
        filt = '('+'|'.join(escape(stat) for stat in ev['stat'])+')'
        filt += f'_{date.strftime("%Y%m%d")}_'
        if start.hour == end.hour:
            filt += f'{start.hour:02d}'+'('+'|'.join(
                map(toregex, _digit_range(max(0, 15*(start.minute//15)-2), end.minute))
            )+')'
        else:
            mins = [f'{h:02d}([0-5][0-9])' for h in range(start.hour, end.hour+1)]
            mins[0] = f'{start.hour:02d}'+'('+'|'.join(
                map(toregex, _digit_range(max(0, 15*(start.minute//15)-2), 59))
            )+')'
            mins[-1] = f'{end.hour:02d}'+'('+'|'.join(
                map(toregex, _digit_range(0, end.minute))
            )+')'
            filt += '('+'|'.join(mins)+')'
        filt += r'[0-9]{2}_[0-9]{2}.*'
        args = f'--accept gz --accept-regex "{filt}" {DATAURL}/2002-20yy_Callisto/{date.strftime("%Y/%m/%d")}'
        prefix = f' -P {path!r} '
        run(shlex.split(wget_default+prefix+args), check=True)

def read_list(path):
    line = lambda l: (l.strip() if l[0].isnumeric()
        and '##:' not in l
        and '??:' not in l
        and '§' not in l
        and '1447' not in l
        and '6:88' not in l
        and '24:3' not in l
        and '212:' not in l
        and len(l.split('\t')) > 2
        else '')
    with open(path, 'r', encoding='latin-1') as f:
        sio = io.StringIO('\n'.join(map(line, f.readlines())))
        df = pd.read_csv(
            sio,
            sep='\t',
            comment='#',
            on_bad_lines='warn',
            names=('date', 'time', 'type', 'stat'),
            converters={
                'date': lambda s: pd.to_datetime(s, format='%Y%m%d'),
                'time': lambda s: tuple(map(
                    lambda sp: pd.to_datetime(f'{"00" if sp[0] == "24" else sp[0]} {sp[1]}', format='%H %M'),
                    zip(*[iter(s.translate({ord(c): ':' for c in '-_.'}).strip('+').split(':'))]*2),
                )),
                'type': lambda s: s.strip(),
                'stat': lambda s: set(map(lambda s: s.strip(), s.strip(',').split(','))),
            },
        )
    return df

def load_lists(path=DATADIR):
    fpath = os.path.join(path, LISTDIR)
    if not os.path.exists(fpath): get_lists(path)
    lists = sorted(glob(os.path.join(fpath, '**', '*.txt'), recursive=True))
    dfs = reduce(
        lambda *a: pd.concat(a, ignore_index=True, copy=False),
        filter(lambda df: not df.empty, (read_list(fp) for fp in lists)),
    )
    dfs = dfs.loc[dfs['time'].map(lambda t: t[0] <= t[1])]
    dfs['time'] = dfs['time'].map(lambda t: Interval(
        t[0], t[1]+Timedelta(seconds=59),
        closed='both',
    ))
    return dfs

def load_fits(blist, path=DATADIR):
    fpath = os.path.join(path, FITSDIR)
    if not os.path.exists(fpath): get_fits(path, blist)
    lists = map(
        lambda s: (s, *s.split(os.path.sep)[-1].strip('.fit.gz').rsplit('_', 3)[-3:]),
        sorted(glob(os.path.join(fpath, '**', '*.fit.gz'), recursive=True)),
    )
    dff = pd.DataFrame(lists, columns=['path', 'date', 'time', 'fcode'])
    dff['date'] = pd.to_datetime(dff['date'], format='%Y%m%d')
    dff['time'] = dff['time'].map(lambda s: Interval(
        pd.to_datetime(s, format='%H%M%S'),
        pd.to_datetime(s, format='%H%M%S')+Timedelta(minutes=14, seconds=59),
        closed='both',
    ))
    dff['evnt'] = dff.apply(
        lambda r: blist.loc[(blist['date'] == r['date']) & blist['time'].array.overlaps(r['time'])].index.tolist(),
        axis=1,
    )
    dff = dff.loc[dff['evnt'].map(bool)]
    return dff

def select_dataset(
    blist,
    selectors=sel_default,
    type_cnt_min=20,
    stat_cnt_min=20,
    evnt_dur_max='01:00:00',
    date_start='2020-03-08',
    date_end='2023-09-15',
):
    sel = blist.explode('stat')
    sel['type'] = sel['type'].map(selectors['typem'])
    sel['stat'] = sel['stat'].map(selectors['statm'])
    mask = (sel['date'] >= Timestamp.fromisoformat(date_start)) \
        & (sel['date'] <= Timestamp.fromisoformat(date_end)) \
        & sel['type'].map(selectors['typef']) \
        & sel['stat'].map(selectors['statf'])
    sel = sel.loc[mask]
    cfreq = sel['type'].value_counts()
    sfreq = sel['stat'].value_counts()
    dur = sel['time'].map(lambda t: t.length)
    mask = sel['type'].isin(cfreq[cfreq >= type_cnt_min].index) \
        & sel['stat'].isin(sfreq[sfreq >= stat_cnt_min].index) \
        & (dur >= Timedelta(0)) & (dur <= Timedelta(evnt_dur_max))
    sel = sel.loc[mask]
    sel = sel.groupby([sel.index, 'date', 'time', 'type'])['stat'].apply(list).reset_index()
    sel['type'] = sel['type'].astype('category')
    return sel
