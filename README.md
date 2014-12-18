<link href="//maxcdn.bootstrapcdn.com/font-awesome/4.1.0/css/font-awesome.min.css" rel="stylesheet" />

<i class="fa fa-file" style="font-size:1em;"></i>卒論
========

卒論に向けた試行錯誤

## Description

- はじめに([sim1](simple1/sim1.ipynb))


- 簡単な例1([simple1](simple1/simple1.ipynb))
    - done
    - 条件
        - 沈黙なし
        - 意見は$[0,1]$の一様乱数(重要)
        - 発言者を選ぶ確率は等しい
    - 結果
        - 発言がなされたときにリンクが張られる数は二項分布$B(k,p(x_{k+1},r))$に従う
        - $y$を$k,r$にしたがって変化する適当な関数におけば、リンクの個数が$y$個を超えるのに必要な時間は幾何分布に従う


- 簡単な例2([simple2](simple2/simple2.ipynb))
    - done
    - 条件
        - 沈黙あり([0,1]の間の一様乱数の期待値0.5)
        - 意見は$[0,1]$の一様乱数(重要)
        - 発言者を選ぶ確率は距離の関数
    - 結果
        - 位置によって発言者の発現頻度などは異なる(当たり前)
        - リンクの数等についての性質は例1と同じ(当たり前)
        - 各時刻$k$において張られるリンクの数の$r$依存性などに対する計算


- 参加者間のネットワークについて([person_network](simple2/person_network.ipynb))
    - done
    - 参加者間のネットワークのみを考えると、(結局)マルコフ連鎖でかける


- 参加者間のネットワークについて(2)([person_network2](simple2/person_network2.ipynb))
    - 途中 & 意味はない
    - 距離のみで確率が決まるような場合


- 意見の作るネットワークについて([idea](simple1/idea.ipynb))
    - 途中
    - 意見の分布に関する観察をする予定


- 簡単な例3
    - 例1、2とは異なったアルゴリズムでの会議のシミュレーション
    - 途中
    - 条件
        - 参加者の意見の数は有限
        - 意見によって発言者が決まるとする
        - 意見間のリンクは考えていない
    - 以前の意見が次の発言の意見にどう影響するかで場合分け
        - [simple3_123](simple3/simple3_123.ipynb)
            - １. 影響なし(独立)
            - ２. 議題
            - ３. 一つ前の意見
        - [simple3_45](simple3/simple3_45.ipynb)
            - ４. 一つ前の意見 + 議題
            - ５. 二つ前までの意見
        - ６. 二つ前までの意見 + 議題
        - ７. それ以上前の意見
    - 結果
        - いくつかのパラメータの効果を観察によって定性的に理解した

## Author

[ssh0](https://github.com/ssh0)


