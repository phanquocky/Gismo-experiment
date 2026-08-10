# Identifying Code và Set Cover Greedy

Tài liệu này tóm tắt cách chuyển bài toán **Identifying Code** thành **Set Cover** và áp dụng thuật toán **Greedy Set Cover** để tìm một identifying code xấp xỉ, dựa trên paper:

> M. Laifenfeld, A. Trachtenberg, and T. Y. Berger-Wolf,  
> *Identifying Codes and the Set Cover Problem*, Allerton Conference, 2006.

---

## 1. Bài toán Identifying Code trên đồ thị vô hướng

Cho một **đồ thị vô hướng**

\[
G=(V,E).
\]

Với mỗi đỉnh \(v\in V\), ký hiệu **lân cận đóng** của \(v\) là

\[
N[v]
=
\{v\}\cup\{u\in V\mid \{u,v\}\in E\}.
\]

Cho một tập đỉnh

\[
C\subseteq V,
\]

ta gọi \(C\) là một **code**, và các phần tử của \(C\) là các **codeword**.

### Identifying set

Identifying set của một đỉnh \(v\) đối với code \(C\) được định nghĩa là

\[
I_C(v)=N[v]\cap C.
\]

Tập \(C\) là một **identifying code** nếu mỗi đỉnh có một identifying set khác rỗng và khác với identifying set của mọi đỉnh còn lại. Cụ thể:

### Điều kiện domination

Với mọi \(v\in V\),

\[
I_C(v)=N[v]\cap C\neq\varnothing.
\]

Điều này đảm bảo mỗi đỉnh được quan sát bởi ít nhất một codeword.

### Điều kiện separation

Với mọi hai đỉnh phân biệt \(u,v\in V\),

\[
u\neq v
\quad\Longrightarrow\quad
I_C(u)\neq I_C(v).
\]

Như vậy, mỗi đỉnh được xác định duy nhất bởi tập các codeword nằm trong lân cận đóng của nó.

### Bài toán tối ưu

Bài toán **Minimum Identifying Code** yêu cầu tìm identifying code có số lượng đỉnh nhỏ nhất:

\[
\min_{C\subseteq V}|C|
\]

sao cho

\[
N[v]\cap C\neq\varnothing,
\qquad
\forall v\in V,
\]

và

\[
N[u]\cap C
\neq
N[v]\cap C,
\qquad
\forall u\neq v.
\]

---

## 2. Bài toán Set Cover

Cho một tập nền

\[
U=\{u_1,u_2,\ldots,u_m\}
\]

và một họ các tập con

\[
\mathcal S=\{S_1,S_2,\ldots,S_k\},
\qquad
S_i\subseteq U.
\]

Một tập con

\[
\mathcal C\subseteq \mathcal S
\]

được gọi là một **set cover** nếu

\[
\bigcup_{S\in\mathcal C}S=U.
\]

Bài toán **Minimum Set Cover** yêu cầu tìm số lượng tập con nhỏ nhất để phủ toàn bộ \(U\):

\[
\min_{\mathcal C\subseteq\mathcal S}|\mathcal C|
\]

sao cho

\[
\bigcup_{S\in\mathcal C}S=U.
\]

---

## 3. Chuyển Identifying Code thành Set Cover

Ý tưởng chính của paper là xem mỗi yêu cầu

\[
\text{“phân biệt hai đỉnh }u\text{ và }v\text{”}
\]

như một **phần tử cần được cover**.

### 3.1. Universe của bài toán Set Cover

Với đồ thị \(G=(V,E)\), xây dựng universe

\[
U_G=
\{(u,v)\mid u,v\in V,\;u\neq v\}.
\]

Nếu không phân biệt thứ tự của hai đỉnh thì universe có

\[
|U_G|=\frac{n(n-1)}{2}
\]

phần tử.

Mỗi phần tử

\[
(u,v)\in U_G
\]

biểu diễn một constraint:

> Hai đỉnh \(u\) và \(v\) phải được phân biệt bởi identifying code.

Với định nghĩa identifying code trên đồ thị vô hướng, ta cũng yêu cầu domination

\[
N[v]\cap C\neq\varnothing
\qquad \forall v\in V.
\]

Trong formulation của paper, điều kiện phân biệt được biểu diễn thông qua các cặp đỉnh và các distinguishing set. Khi triển khai một solver cho IC theo định nghĩa chuẩn, cần đồng thời kiểm tra điều kiện domination ở trên.

---

### 3.2. Difference set

Với hai đỉnh \(u,v\), paper định nghĩa **difference set**

\[
D_C(u,v)
=
I_C(u)\triangle I_C(v),
\]

trong đó \(\triangle\) là symmetric difference.

Với toàn bộ tập đỉnh \(V\) được xem như code, ta viết

\[
D(u,v)
=
N[u]\triangle N[v].
\]

Một đỉnh \(c\) có thể phân biệt \(u\) và \(v\) khi và chỉ khi

\[
c\in D(u,v).
\]

---

### 3.3. Distinguishing set của một đỉnh

Với mỗi candidate codeword

\[
c\in V,
\]

ta tạo một tập

\[
\delta_c
=
\{(u,v)\in U_G\mid c\in D(u,v)\}.
\]

Ý nghĩa:

\[
(u,v)\in\delta_c
\]

khi và chỉ khi việc chọn \(c\) vào code giúp phân biệt \(u\) và \(v\).

Do đó mỗi vertex \(c\) của đồ thị tương ứng với một set \(\delta_c\) trong bài toán Set Cover.

Ta thu được họ tập

\[
\Delta
=
\{\delta_c\mid c\in V\}.
\]

---

## 4. Quan hệ giữa Identifying Code và Set Cover

Paper chứng minh rằng:

\[
\boxed{
C\text{ là identifying code}
\iff
\{\delta_c\mid c\in C\}
\text{ cover }U_G
}
\]

hay

\[
\boxed{
\bigcup_{c\in C}\delta_c
=
U_G.
}
\]

Do đó bài toán Minimum Identifying Code trở thành:

\[
\min_{C\subseteq V}|C|
\]

sao cho

\[
\bigcup_{c\in C}\delta_c
=
U_G.
\]

Đây chính là một instance của **Minimum Set Cover**.

Có thể nhìn phép biến đổi như sau:

\[
\boxed{
\text{vertex }c
}
\quad\longrightarrow\quad
\boxed{
\text{set }\delta_c
}
\]

và

\[
\boxed{
\text{pair }(u,v)
}
\quad\longrightarrow\quad
\boxed{
\text{element cần cover}
}.
\]

---

# 5. Áp dụng Greedy Set Cover để giải Identifying Code

Sau khi chuyển bài toán IC thành Set Cover, ta có thể áp dụng trực tiếp thuật toán **Greedy Set Cover**.

## Greedy Set Cover

Tại mỗi bước, chọn tập phủ được nhiều phần tử chưa được phủ nhất.

Trong bài toán Identifying Code, điều này tương đương với:

> Chọn vertex phân biệt được nhiều cặp đỉnh chưa được phân biệt nhất.

Giả sử \(R\subseteq U_G\) là tập các pair chưa được cover.

Tại mỗi vòng lặp, chọn

\[
c^*
=
\arg\max_{c\in V\setminus C}
|\delta_c\cap R|.
\]

Sau đó cập nhật

\[
C
\leftarrow
C\cup\{c^*\}
\]

và

\[
R
\leftarrow
R\setminus\delta_{c^*}.
\]

Lặp lại cho đến khi

\[
R=\varnothing.
\]

---

## 6. Thuật toán ID-Greedy

Theo Construction 2 trong paper:

### Input

Đồ thị

\[
G=(V,E).
\]

### Output

Một identifying code \(C_{\text{greedy}}\).

### Các bước

1. Tính identifying set của tất cả các đỉnh:

\[
I(v)=N[v].
\]

2. Tính distinguishing set của mỗi đỉnh:

\[
\delta_c
=
\{(u,v)\in U_G\mid c\in D(u,v)\}.
\]

3. Xây dựng Set Cover instance

\[
(U_G,\Delta),
\]

với

\[
\Delta=\{\delta_c\mid c\in V\}.
\]

4. Chạy thuật toán **Greedy Set Cover** trên

\[
(U_G,\Delta).
\]

5. Nếu greedy chọn các set

\[
\delta_{c_1},\delta_{c_2},\ldots,\delta_{c_k},
\]

thì trả về

\[
C_{\text{greedy}}
=
\{c_1,c_2,\ldots,c_k\}.
\]

---

## 7. Pseudocode

```text
ID-GREEDY(G):

    U = {(u,v) | u,v ∈ V, u ≠ v}

    for each c ∈ V:
        δ[c] = ∅

        for each (u,v) ∈ U:
            if c ∈ B(u) △ B(v):
                δ[c] = δ[c] ∪ {(u,v)}

    C = ∅
    R = U

    while R ≠ ∅:
        c* = argmax_c |δ[c] ∩ R|

        C = C ∪ {c*}
        R = R \ δ[c*]

    return C
```

---

## 8. Approximation guarantee

Với đồ thị có \(n\) đỉnh, Set Cover instance có universe kích thước

\[
m=
\frac{n(n-1)}{2}.
\]

Greedy Set Cover có approximation ratio logarithmic theo kích thước universe.

Do đó:

\[
O(\log m)
=
O\left(
\log \frac{n(n-1)}{2}
\right)
=
O(\log n).
\]

Paper chứng minh cụ thể rằng tồn tại các hằng số \(c_1,c_2\ge 0\) sao cho

\[
c_1\ln n
<
\frac{|C_{\text{greedy}}|}
{|C_{\min}|}
<
c_2\ln n.
\]

Do đó:

\[
\boxed{
\frac{|C_{\text{greedy}}|}
{|C_{\min}|}
=
\Theta(\log n)
}
\]

trong trường hợp tổng quát.

---

## 9. Ý tưởng cốt lõi

Toàn bộ phương pháp có thể tóm tắt bằng sơ đồ:

\[
G
\]

\[
\Downarrow
\]

\[
\text{các constraint }(u,v)
\]

\[
\Downarrow
\]

\[
U_G=\{(u,v):u\neq v\}
\]

và với mỗi vertex \(c\),

\[
\delta_c
=
\{\text{các constraint mà }c\text{ giải quyết}\}.
\]

Sau đó:

\[
\boxed{
\text{Identifying Code}
}
\quad\longrightarrow\quad
\boxed{
\text{Minimum Set Cover}
}
\]

và áp dụng:

\[
\boxed{
\text{Greedy Set Cover}
}
\quad\longrightarrow\quad
\boxed{
C_{\text{greedy}}.
}
\]

Ý nghĩa trực quan nhất là:

> Mỗi cặp đỉnh \((u,v)\) là một constraint cần được giải quyết.  
> Mỗi vertex \(c\) giải quyết tất cả các constraint mà nó có thể phân biệt.  
> Ta chọn ít vertex nhất sao cho mọi constraint đều được giải quyết.

---

## Tài liệu tham khảo

M. Laifenfeld, A. Trachtenberg, and T. Y. Berger-Wolf,  
**“Identifying Codes and the Set Cover Problem,”**  
44th Annual Allerton Conference on Communication, Control, and Computing, 2006.
