"""
데이터 전처리를 위한 모듈

이 모듈은 대화 요약 데이터셋을 전처리하고 모델 입력 형태로 가공하는 Preprocess 클래스를 제공합니다.
"""

import pandas as pd
from typing import Tuple, List, Union


class Preprocess:
    """
    데이터 전처리를 위한 클래스로, 데이터셋을 데이터프레임으로 변환하고 인코더와 디코더의 입력을 생성합니다.

    Attributes:
        bos_token (str): Beginning of Sequence 토큰 (디코더 입력의 시작 토큰)
        eos_token (str): End of Sequence 토큰 (디코더 출력의 종료 토큰)
    """

    def __init__(
        self,
        bos_token: str,
        eos_token: str,
    ) -> None:
        """
        Preprocess 클래스 초기화

        Args:
            bos_token (str): Beginning of Sequence 토큰
            eos_token (str): End of Sequence 토큰
        """
        self.bos_token = bos_token
        self.eos_token = eos_token

    @staticmethod
    def make_set_as_df(file_path: str, is_train: bool = True) -> pd.DataFrame:
        """
        CSV 파일에서 실험에 필요한 컬럼을 가져와 데이터프레임으로 반환합니다.

        Args:
            file_path (str): CSV 파일 경로
            is_train (bool): 학습 데이터 여부 (True: train/validation, False: test)
                           True인 경우 fname, dialogue, summary 컬럼을 반환
                           False인 경우 fname, dialogue 컬럼만 반환

        Returns:
            pd.DataFrame: 필요한 컬럼만 포함된 데이터프레임
        """
        if is_train:
            df = pd.read_csv(file_path)
            train_df = df[['fname', 'dialogue', 'summary']]
            return train_df
        else:
            df = pd.read_csv(file_path)
            test_df = df[['fname', 'dialogue']]
            return test_df

    def make_input(
        self,
        dataset: pd.DataFrame,
        is_test: bool = False
    ) -> Union[Tuple[List[str], List[str]], Tuple[List[str], List[str], List[str]]]:
        """
        BART 모델의 입력, 출력 형태를 맞추기 위해 전처리를 진행합니다.

        Args:
            dataset (pd.DataFrame): 전처리할 데이터프레임
            is_test (bool): 테스트 데이터 여부
                          True인 경우: (encoder_input, decoder_input) 반환
                          False인 경우: (encoder_input, decoder_input, decoder_output) 반환

        Returns:
            Union[Tuple[List[str], List[str]], Tuple[List[str], List[str], List[str]]]:
                is_test=True: (encoder_input, decoder_input)
                    - encoder_input: 대화문 리스트
                    - decoder_input: bos_token 리스트
                is_test=False: (encoder_input, decoder_input, decoder_output)
                    - encoder_input: 대화문 리스트
                    - decoder_input: bos_token + summary 리스트 (Ground truth를 디코더의 input으로 사용하여 학습)
                    - decoder_output: summary + eos_token 리스트 (학습 시 예측해야 할 정답)
        """
        if is_test:
            encoder_input = dataset['dialogue']
            decoder_input = [self.bos_token] * len(dataset['dialogue'])
            return encoder_input.tolist(), list(decoder_input)
        else:
            encoder_input = dataset['dialogue']
            # Ground truth를 디코더의 input으로 사용하여 학습합니다 (Teacher Forcing)
            decoder_input = dataset['summary'].apply(lambda x: self.bos_token + str(x))
            decoder_output = dataset['summary'].apply(lambda x: str(x) + self.eos_token)
            return encoder_input.tolist(), decoder_input.tolist(), decoder_output.tolist()
