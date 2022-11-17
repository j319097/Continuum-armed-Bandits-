# GaussianProcess

ガウス過程

線形回帰モデル

$$
\hat{y} = w_0\phi_0(X) + w_1\phi_1(X)+w_2\phi_2(X)+ \ldots + w_H\phi_H(X)\\
$$

を縦にN個並べることで

$$
\begin{pmatrix}
\hat{y_1} \\
\hat{y_2} \\
\vdots \\
\hat{y_N} \\ 
\end{pmatrix}
\ = 
\begin{pmatrix}
\phi_0(X_1) \quad  \phi_1(X_1) \quad  \cdots \quad \phi_H(X_1) \\
\phi_0(X_2) \quad  \phi_1(X_2) \quad  \cdots \quad \phi_H(X_2) \\
\vdots       \qquad  \qquad \qquad  \qquad   \qquad  \vdots \\
\phi_0(X_N) \quad  \phi_1(X_N) \quad  \cdots \quad \phi_H(X_N) \\
\end{pmatrix}
\times
\begin{pmatrix}
w_1 \\
w_2 \\
\vdots \\
w_H \\ 
\end{pmatrix}\\
$$

$$
  \Phi_nh = \phi_h(X_n)を要素とする計画行列\Phiを使って\\
  \hat{Y} = \Phi W\\
$$

と書くことができる

$$
  \PhiはX_1,...,X_Nが与えられれば定数行列になる
$$

$$
  \hat{Y} = Yとする\\
  つまり,Y = \Phi Wとなると仮定する\\
  さらに,重みWが\\\
  W\sim N(0,\lambda^2I)から生成されると仮定する。\\
$$

整理すると

$$
  Y が「ガウス分布に従うWを定数行列\Phiで線形変換したもの」\\
  であることを意味する
$$

期待値は

$$
  E[Y]=E[\Phi W]= \Phi E[W] = 0から
  \Sigma  = E[YY^T] - E[Y]E[Y]^T = E[(\Phi W)(\Phi W)^T] = \Phi E[WW^T]\Phi^T \\
$$

$$  
  (= λ^2\Phi \Phi ^T
  E[WW^T] = λIであることを用いた)
$$

結果として

$$
  Y \sim N(0,\lambda^2\Phi \Phi^T)\\
$$

となることがわかる
共分散行列を

$$
K = \lambda^2\Phi \Phi^T  = \lambda^2
\begin{pmatrix}
\vdots \\
\phi(X_n)^T \\ 
\vdots \\
\end{pmatrix}
\begin{pmatrix}
\cdots \phi(X_{n'}) \cdots
\end{pmatrix}
$$

とおくと

$$
  K = \lambda^2\Phi \Phi^T\\
  K_{nn'} = \lambda^2 \phi(X_n)^T \phi (X_{n'})\\
$$

が与えられる








