"""
PyTorch Dataset 클래스를 정의하는 모듈

이 모듈은 학습(Train), 검증(Validation), 추론(Inference)을 위한 PyTorch Dataset 클래스를 제공합니다.
"""

import torch
from torch.utils.data import Dataset
from typing import Dict, List, Any


class DatasetForTrain(Dataset):
    """
    학습(Train)에 사용되는 Dataset 클래스

    BART 모델 학습을 위한 encoder input, decoder input, labels를 포함하는 데이터셋입니다.
    Teacher Forcing 방식으로 학습하기 위해 decoder input과 labels를 별도로 관리합니다.

    Attributes:
        encoder_input (Dict[str, torch.Tensor]): 인코더 입력 (input_ids, attention_mask)
        decoder_input (Dict[str, torch.Tensor]): 디코더 입력 (input_ids, attention_mask)
        labels (Dict[str, torch.Tensor]): 정답 레이블 (input_ids)
        len (int): 데이터셋 크기
    """

    def __init__(
        self,
        encoder_input: Dict[str, torch.Tensor],
        decoder_input: Dict[str, torch.Tensor],
        labels: Dict[str, torch.Tensor],
        len: int
    ):
        """
        DatasetForTrain 초기화

        Args:
            encoder_input (Dict[str, torch.Tensor]): 토크나이징된 인코더 입력
            decoder_input (Dict[str, torch.Tensor]): 토크나이징된 디코더 입력
            labels (Dict[str, torch.Tensor]): 토크나이징된 정답 레이블
            len (int): 데이터셋 크기
        """
        self.encoder_input = encoder_input
        self.decoder_input = decoder_input
        self.labels = labels
        self.len = len

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        주어진 인덱스에 해당하는 데이터 샘플을 반환합니다.

        Args:
            idx (int): 데이터 인덱스

        Returns:
            Dict[str, torch.Tensor]: 다음 키를 포함하는 딕셔너리
                - input_ids: 인코더 입력 토큰 ID
                - attention_mask: 인코더 어텐션 마스크
                - decoder_input_ids: 디코더 입력 토큰 ID
                - decoder_attention_mask: 디코더 어텐션 마스크
                - labels: 정답 토큰 ID
        """
        # 인코더 입력 (input_ids, attention_mask)
        item = {key: val[idx].clone().detach() for key, val in self.encoder_input.items()}

        # 디코더 입력 (input_ids, attention_mask)
        item2 = {key: val[idx].clone().detach() for key, val in self.decoder_input.items()}

        # 디코더 입력의 키 이름 변경
        item2['decoder_input_ids'] = item2['input_ids']
        item2['decoder_attention_mask'] = item2['attention_mask']
        item2.pop('input_ids')
        item2.pop('attention_mask')

        # 인코더와 디코더 입력 병합
        item.update(item2)

        # 정답 레이블 추가
        item['labels'] = self.labels['input_ids'][idx]

        return item

    def __len__(self) -> int:
        """
        데이터셋 크기를 반환합니다.

        Returns:
            int: 데이터셋 크기
        """
        return self.len


class DatasetForVal(Dataset):
    """
    검증(Validation)에 사용되는 Dataset 클래스

    BART 모델 검증을 위한 encoder input, decoder input, labels를 포함하는 데이터셋입니다.
    구조는 DatasetForTrain과 동일하지만, 검증 시 다른 동작이 필요할 경우를 대비하여 별도 클래스로 구현합니다.

    Attributes:
        encoder_input (Dict[str, torch.Tensor]): 인코더 입력 (input_ids, attention_mask)
        decoder_input (Dict[str, torch.Tensor]): 디코더 입력 (input_ids, attention_mask)
        labels (Dict[str, torch.Tensor]): 정답 레이블 (input_ids)
        len (int): 데이터셋 크기
    """

    def __init__(
        self,
        encoder_input: Dict[str, torch.Tensor],
        decoder_input: Dict[str, torch.Tensor],
        labels: Dict[str, torch.Tensor],
        len: int
    ):
        """
        DatasetForVal 초기화

        Args:
            encoder_input (Dict[str, torch.Tensor]): 토크나이징된 인코더 입력
            decoder_input (Dict[str, torch.Tensor]): 토크나이징된 디코더 입력
            labels (Dict[str, torch.Tensor]): 토크나이징된 정답 레이블
            len (int): 데이터셋 크기
        """
        self.encoder_input = encoder_input
        self.decoder_input = decoder_input
        self.labels = labels
        self.len = len

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        주어진 인덱스에 해당하는 데이터 샘플을 반환합니다.

        Args:
            idx (int): 데이터 인덱스

        Returns:
            Dict[str, torch.Tensor]: 다음 키를 포함하는 딕셔너리
                - input_ids: 인코더 입력 토큰 ID
                - attention_mask: 인코더 어텐션 마스크
                - decoder_input_ids: 디코더 입력 토큰 ID
                - decoder_attention_mask: 디코더 어텐션 마스크
                - labels: 정답 토큰 ID
        """
        # 인코더 입력 (input_ids, attention_mask)
        item = {key: val[idx].clone().detach() for key, val in self.encoder_input.items()}

        # 디코더 입력 (input_ids, attention_mask)
        item2 = {key: val[idx].clone().detach() for key, val in self.decoder_input.items()}

        # 디코더 입력의 키 이름 변경
        item2['decoder_input_ids'] = item2['input_ids']
        item2['decoder_attention_mask'] = item2['attention_mask']
        item2.pop('input_ids')
        item2.pop('attention_mask')

        # 인코더와 디코더 입력 병합
        item.update(item2)

        # 정답 레이블 추가
        item['labels'] = self.labels['input_ids'][idx]

        return item

    def __len__(self) -> int:
        """
        데이터셋 크기를 반환합니다.

        Returns:
            int: 데이터셋 크기
        """
        return self.len


class DatasetForInference(Dataset):
    """
    추론(Inference)에 사용되는 Dataset 클래스

    BART 모델 추론을 위한 encoder input과 test ID를 포함하는 데이터셋입니다.
    추론 시에는 decoder input과 labels가 필요하지 않으므로 encoder input만 사용합니다.

    Attributes:
        encoder_input (Dict[str, torch.Tensor]): 인코더 입력 (input_ids, attention_mask)
        test_id (List[str]): 테스트 데이터 식별자 (fname)
        len (int): 데이터셋 크기
    """

    def __init__(
        self,
        encoder_input: Dict[str, torch.Tensor],
        test_id: List[str],
        len: int
    ):
        """
        DatasetForInference 초기화

        Args:
            encoder_input (Dict[str, torch.Tensor]): 토크나이징된 인코더 입력
            test_id (List[str]): 테스트 데이터 식별자 리스트
            len (int): 데이터셋 크기
        """
        self.encoder_input = encoder_input
        self.test_id = test_id
        self.len = len

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        주어진 인덱스에 해당하는 데이터 샘플을 반환합니다.

        Args:
            idx (int): 데이터 인덱스

        Returns:
            Dict[str, Any]: 다음 키를 포함하는 딕셔너리
                - input_ids: 인코더 입력 토큰 ID
                - attention_mask: 인코더 어텐션 마스크
                - ID: 테스트 데이터 식별자
        """
        # 인코더 입력 (input_ids, attention_mask)
        item = {key: val[idx].clone().detach() for key, val in self.encoder_input.items()}

        # 테스트 데이터 식별자 추가
        item['ID'] = self.test_id[idx]

        return item

    def __len__(self) -> int:
        """
        데이터셋 크기를 반환합니다.

        Returns:
            int: 데이터셋 크기
        """
        return self.len
