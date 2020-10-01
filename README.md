# Tello Control

# インストール

Anacondaがインストールされている前提とする。

pipからtellopyをインストール

```console
$ pip install tellopy
```

Anacondaから必要なライブラリをインストール

```console
$ conda install numpy matplotlib pillow
$ conda install -c conda-forge scipy av opencv
```

Note: PyAVはFFmpegのPythonバインディング

# ファイル構成


| スクリプト | 内容 |
|-----------|------|
| create_calibration_board.py | カメラキャリブレーション用のCharuco board作成 |
| calibrate_from_images.py | キャリブレーション用画像から補正パラメータ算出 |
| create_marker_map.py | 位置推定用のマーカー作成 |
| control.py | 制御用のメインスクリプト |
| face_recognition.py | 表情解析用のメインスクリプト |

# 起動方法

ターミナルを2つ立ち上げる。
1つ目のターミナルでpython face_recognition.pyを実行。
2つ目のターミナルでpython control.pyを実行。

# マニュアル飛行モード

キーボード入力で制御コマンドをTelloに送信する。

```python
# Manual control
#
# q: quit
# z: switch mode between manual and auto
#
#    w           t           i
#  a s d         g         j k l
#
#  pitch      takeoff      vertical
#  & roll    & landing     & yaw
#
```

# 座標系について

![coordinate_systems](https://github.com/kurokis/TelloControl/blob/master/docs/imgs/coordinate_systems.png)

## arucoモジュールが返すrvec, tvecは何を表しているか？

aruco.estimatePoseSingleMarkersが返すrvec, tvecは何を表現しているか？
カメラとマーカーの位置関係を表していることは間違いないから、以下が候補にあがる。

1. カメラからみたマーカーの姿勢と位置をカメラ座標系で記述している
1. カメラからみたマーカーの姿勢と位置をマーカー座標系で記述している
1. マーカーからみたカメラの姿勢と位置をカメラ座標系で記述している
1. マーカーからみたカメラの姿勢と位置をマーカー座標系で記述している

このうち2.と3.は移動の起点と記述する座標系が異なるから不自然ではあるが、候補に残しておく。

なお、上記はいずれも空間内で座標軸の位置を移動させる意味合いで書いており、これをactive transformationという。一方で、空間上に固定された点のxyz成分表記が座標系変換でどう変わるかという見方もでき、これをpassive transformationという。Passive transformationも加えるとさらに2つの解釈が生まれる。

5. カメラ座標系でみた点をマーカー座標系で見たときの変換式を与える
5. マーカー座標系でみた点をカメラ座標系で見たときの変換式を与える

結論としてはaruco.estimatePoseSingleMarkersは1を表現している。

## どうやって検証する？

検証するにしても、そもそもどのような表現方法で回転を記載するかを決めなければならない。
回転の記述方法にはオイラー角、回転行列、ロドリゲス、クオータニオンなどの方法があり一長一短である。相互変換はScipy等のライブラリで提供されているから簡単だが、使い方をよく確認しないと仮説が間違っているのかコードが間違っているのか区別がつかなくなるため注意が必要である。


### オイラー角

オイラー角は定義の曖昧性に注意が必要である。
オイラー角は回転軸の順序の選び方が6通りあり、しかも回転軸の記述方法がintrinsic rotationとextrinsic rotationの2通りある。Intrinsic rotationは逐次回転する座標系の軸を基準に回転方向を記述する方法である。一方、Extrinsic rotationは元の座標系(固定)の軸を基準に回転方向を記述する方法である。

航空分野で通常用いられるオイラー角(nautical angles)はz、y、xの順序のintrinsic rotationで記述したものであり、逐次設定される座標系に'をつけてz-y'-x''のように表現することもある。z、y'、x''軸周りの回転をそれぞれyaw、pitch、rollという。

``scipy.spatial.transform.Rotation.as_euler``の[ドキュメンテーション](https://scipy.github.io/devdocs/generated/scipy.spatial.transform.Rotation.as_euler.html#r72d546869407-1)
を読むと、引数を大文字にするとintrinsic rotation、小文字にするとextrinsic rotationになると書かれている。すなわち``scipy.spatial.transform.rotation.Rotation``オブジェクト``r``があるとき

```python
r.as_euler('zyx', degrees=True)
```
は航空分野でよく使うyaw、pitch、rollを返さず、正しくは
```python
r.as_euler('ZYX', degrees=True)
```
とする必要がある。

### 回転行列

回転行列の各列は元の座標系の基底ベクトルを回転した後のベクトルを与えるから、比較的容易に解釈することができる。

rvecから回転行列への変換は``cv2.Rodrigues``で計算できる。

```python
R, _ = cv2.Rodrigues(rvec)
```

# Z-flipping現象について

ARマーカーによる姿勢推定の基本原理は、画像上に投影されたマーカーの4隅から剛体変換を推定することである。
しかし空間上の4点が同一平面内にある場合、曖昧性が排除できないことが知られている。
これをARマーカーに限らず一般のPnP問題に内在する課題であり、z-flipping現象などと呼ばれることがある。

[Improper or bad rotation estimation with solvePnP in some cases](https://github.com/opencv/opencv/issues/8813)


最もシンプルな対処法は、複数のマーカーを同一平面上に乗らないよう配置することである。
例えば下記のようにマーカーを3次元的に配置すればよい。

![marker_board](https://github.com/kurokis/TelloControl/blob/master/docs/imgs/marker_board.jpg)
