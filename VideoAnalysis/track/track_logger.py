import csv
import os
from pathlib import Path

from ..utils import get_pylogger
LOGGER = get_pylogger(__name__)


class TrackLogger:
    def __init__(self, cfg):
        self.cfg = cfg
        self.__check_mark_folder_exist()
    

    def __check_mark_folder_exist(self):
        """Checking saving path
        if already exists, create new default marker path
        if not, create maker path
        """
        marker_path = Path(self.cfg.COMMON.MARKER_PATH)

        if not marker_path.exists():
            marker_path.mkdir(parents=True, exist_ok=False)
        else:
            LOGGER.info(f'{marker_path} is already exist!!!. Making default mark folder...')
            
            # __path = os.path.abspath(os.getcwd()).split(os.sep)[:-2]
            __path = self.cfg.COMMON.MARKER_PATH.split(os.sep)[:-1]
            __path = Path(*__path)

            if not __path.exists():
                incremental_path = __path / 'default1'
            else:
                # check incremental path
                for i in range(1, 9999):
                    incremental_path = __path / f'default{i}'
                    if not incremental_path.exists():
                        break

            incremental_path.mkdir(parents=True, exist_ok=False)
            self.cfg.COMMON.MARKER_PATH = incremental_path.as_posix()

        LOGGER.info(f'Saving result to {self.cfg.COMMON.MARKER_PATH}')
    

    def logging_common_infor(self, time_start_end_infor):
        """Saving common information:
        - Video FPS
        - Number of objects 
        - time start and time end of each object

        Parameters:
        -----------
            time_start_end_infor, dict:
                store time_start_end information of source video
        """
        mark_folder = self.cfg.COMMON.MARKER_PATH 
        common_csv = Path(mark_folder, 'common.csv')

        with common_csv.open('w') as f:
            csv_writer = csv.writer(f, delimiter=' ')
            for k, v in sorted(time_start_end_infor.items()): 
                    csv_writer.writerow([k, v[0], v[1]])
             