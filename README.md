<script type="text/x-mathjax-config">
  MathJax.Hub.Config({ tex2jax: { inlineMath: [['$','$'], ["\\(","\\)"]] } });
</script>
<script type="text/javascript"
  src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS_HTML">
</script>
<meta http-equiv="X-UA-Compatible" CONTENT="IE=EmulateIE7" />


卒論
========

## Description

卒論に向けた試行錯誤

- はじめに([sim1](simple1/sim1.ipynb))

- 簡単な例1([simple1](simple1/simple1.ipynb))
    - 条件
        - 沈黙なし
        - 意見は$[0,1]$の一様乱数(重要)
        - 発言者を選ぶ確率は等しい
    - 結果
        - 発言がなされたときにリンクが張られる数は二項分布$B(k,p(x_{k+1},r))$に従う
        - $y$を$k,r$にしたがって変化する適当な関数におけば、リンクの個数が$y$個を超えるのに必要な時間は幾何分布に従う

- 簡単な例2([simple2](simple2/simple2.ipynb))
    - 条件
        - 沈黙あり([0,1]の間の一様乱数の期待値0.5)
        - 意見は$[0,1]$の一様乱数(重要)
        - 発言者を選ぶ確率は距離の関数
    - 結果
        - 位置によって発言者の発現頻度などは異なる(当たり前)
        - リンクの数等についての性質は例1と同じ(当たり前)
        - 各時刻$k$において張られるリンクの数の$r$依存性などに対する計算

- 参加者のつくるネットワーク([person_network](simple2/person_network.ipynb))

<<<<<<< HEAD
- 意見のつくるネットワークについての再考
=======
- 意見のつくるネットワークについての再考()
>>>>>>> fc8226fcc9f69db44d458db65eba9638806c61e5

## Author

[ssh0](https://github.com/ssh0)


