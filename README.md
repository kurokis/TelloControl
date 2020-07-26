# Tello Control

## インストール

Anacondaがインストールされている前提とする。

pipからtellopyをインストール

```console
$ pip install tellopy
```

Anacondaから必要なライブラリをインストール

```console
$ conda install numpy matplotlib
$ conda install -c conda-forge scipy av opencv
```

Note: PyAVはFFmpegのPythonバインディング


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

# 座標変換について

## rvec, tvecは何を表しているか？

aruco.estimatePoseSingleMarkersが返すrvec, tvecは何を表現しているか？
カメラとマーカーの位置関係を表していることは間違いないから、以下の4つが候補である。

1. カメラからみたマーカーの姿勢と位置をカメラ座標系で記述している
1. カメラからみたマーカーの姿勢と位置をマーカー座標系で記述している
1. マーカーからみたカメラの姿勢と位置をカメラ座標系で記述している
1. マーカーからみたカメラの姿勢と位置をマーカー座標系で記述している

このうち2.と3.は移動の起点と記述する座標系が異なるから不自然ではあるが、
先入観を持たずに検証したいため候補に残しておく。

なお、上記はいずれも空間内で座標軸の位置を移動させる意味合いで書いており、これをactive transformationという。一方で、空間上に固定された点のxyz成分表記が座標系変換でどう変わるかという見方もでき、これをpassive transformationという。Passive transformationも加えるとさらに2つの解釈が生まれる。

5. カメラ座標系でみた点をマーカー座標系で見たときの変換式を与える
5. マーカー座標系でみた点をカメラ座標系で見たときの変換式を与える


## どうやって検証する？

検証するにしても、そもそもどのような表現方法で回転を記載するかを決めなければならない。
回転の記述方法にはオイラー角、回転行列、ロドリゲス、クオータニオンなどの方法があり一長一短である。相互変換はScipy等のライブラリで提供されているから簡単だが、使い方をよく確認しないと仮説が間違っているのかコードが間違っているのか区別がつかなくなるため注意が必要である。


### オイラー角
**特にオイラー角は注意が必要**である。
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

