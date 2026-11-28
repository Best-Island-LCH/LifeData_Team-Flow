"""상호작용 변수 만들기 (피처 엔지니어링 페이지 기준).

prep/ 폴더의 인코딩된 데이터(109칸)를 받아 상호작용 칸을 덧붙인다.
전처리 결과 파일은 바꾸지 않고, 모델 비교 노트북에서 "넣은 버전 / 뺀 버전"을
같은 조각으로 비교할 때 이 함수를 불러서 쓴다.

- 행 안의 값끼리 곱하기만 하므로 다른 행의 통계를 쓰지 않는다 (정보 누수 없음).
- 학습용과 평가용에 따로 적용해도 같은 칸이 같은 순서로 생긴다.

사용 예:
    from features import add_interactions
    X_core = add_interactions(X)                          # 확정 묶음만
    X_all  = add_interactions(X, groups=("core", "candidate"))  # 실험 후보까지
"""

import pandas as pd

COUPON_PREFIX = "coupon_"

# 쿠폰과 곱할 상대 변수. 묶음별로 나눈다.
#   onehot  : 원핫 칸(passanger_*, weather_*) → 쿠폰 칸 × 각 범주 칸
#   numeric : 한 칸짜리 정수(expiration 0/1, distance 0/1/2) → 쿠폰 칸 × 그 값
#             (업종마다 그 변수의 효과 크기가 다르게 붙도록)
INTERACTION_GROUPS = {
    # 확정: 업종마다 효과 차이가 크고 근거가 분명한 것
    #   동승자   : 아이 동승 시 바 -19.6%p, 커피숍 +4.3%p
    #   유효기간 : 1d-2h 차이가 저가 식당 +24.0%p, 바 +0.9%p
    "core": [
        ("passanger", "onehot"),
        ("expiration", "numeric"),
    ],
    # 실험 후보: 효과는 있지만 행이 적은 조합이 있는 것
    #   날씨 : 저가 식당 맑음 76.8% vs 비 39.5% / 중가 식당 × 비는 65행뿐
    #   거리 : 저가 식당 25분 이상에서 75.3% → 32.9% / 테이크아웃은 거의 변화 없음
    "candidate": [
        ("weather", "onehot"),
        ("distance", "numeric"),
    ],
}


def _coupon_columns(X: pd.DataFrame) -> list:
    cols = [c for c in X.columns if c.startswith(COUPON_PREFIX)]
    if not cols:
        raise ValueError("coupon_* 원핫 칸이 없습니다. prep/ 폴더의 인코딩된 데이터를 넣어야 합니다.")
    return cols


def _partner_columns(X: pd.DataFrame, name: str, kind: str) -> list:
    if kind == "onehot":
        cols = [c for c in X.columns if c.startswith(name + "_")]
    elif kind == "numeric":
        cols = [name] if name in X.columns else []
    else:
        raise ValueError(f"알 수 없는 종류: {kind}")
    if not cols:
        raise ValueError(f"'{name}' 칸을 찾지 못했습니다.")
    return cols


def interaction_columns(X: pd.DataFrame, groups=("core",)) -> list:
    """만들어질 상호작용 칸 이름 목록 (순서 포함)."""
    names = []
    for g in groups:
        for name, kind in INTERACTION_GROUPS[g]:
            for p in _partner_columns(X, name, kind):
                for c in _coupon_columns(X):
                    names.append(f"{c}__x__{p}")
    return names


def add_interactions(X: pd.DataFrame, groups=("core",)) -> pd.DataFrame:
    """X에 상호작용 칸을 덧붙인 새 DataFrame을 돌려준다 (X는 바꾸지 않음).

    groups: ("core",) 또는 ("core", "candidate") 등 INTERACTION_GROUPS의 키.
    """
    unknown = set(groups) - set(INTERACTION_GROUPS)
    if unknown:
        raise ValueError(f"알 수 없는 묶음: {sorted(unknown)}")

    coupons = _coupon_columns(X)
    new = {}
    for g in groups:
        for name, kind in INTERACTION_GROUPS[g]:
            for p in _partner_columns(X, name, kind):
                for c in coupons:
                    new[f"{c}__x__{p}"] = X[c] * X[p]

    out = pd.concat([X, pd.DataFrame(new, index=X.index)], axis=1)
    if out.columns.duplicated().any():
        raise ValueError("같은 이름의 칸이 생겼습니다. 이미 상호작용 칸이 들어간 X인지 확인하세요.")
    return out
