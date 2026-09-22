# Greedy K=2 không dựng Universe: partition đỉnh gốc và hai loại sensor

## 1. Mục tiêu và quyết định mô hình

Cho đồ thị vô hướng đơn G=(V,E), n=|V|. Có thể có một hoặc hai đỉnh cháy đồng thời. Cần chọn ít sensor nhất để:

1. Phát hiện mọi trạng thái có cháy.
2. Phân biệt mọi hai trạng thái cháy khác nhau, kể cả hai trạng thái có đỉnh chung.

Quyết định triển khai:

- Duy trì partition của **đỉnh gốc** và signature của từng partition.
- Tính gain chính xác bằng tổ hợp và phép OR signature.
- Không dựng các đỉnh gộp, không liệt kê các trạng thái cháy trong thuật toán chính, không dựng Universe các ràng buộc.
- Bổ sung sensor đơn đỉnh để bảo đảm tồn tại nghiệm.
- Mặc định hai loại sensor đều là ứng viên ngay từ đầu, chi phí bằng nhau; tối thiểu hóa tổng số sensor. Greedy là heuristic, không bảo đảm nghiệm tối ưu.

Đây là bài toán mở rộng mô hình sensor của Identifying Code, không còn là bài toán chỉ chọn sensor lân cận đóng truyền thống.

## 2. Hai loại sensor và điều kiện bảo đảm nghiệm

Mỗi sensor a có một tập phát hiện D(a) ⊆ V:

| Loại | ID gợi ý | Tập phát hiện |
|---|---|---|
| Lân cận đóng | (`N`, v) | D(a)=N[v]={v} ∪ {u: {u,v}∈E} |
| Đơn đỉnh | (`L`, v) | D(a)={v} |

Sensor a cho tín hiệu 1 khi và chỉ khi F ∩ D(a) ≠ ∅, với F là tập đỉnh cháy. Tín hiệu của hai đỉnh cháy là OR của tín hiệu từng đỉnh; không đếm số đỉnh cháy.

Tập ứng viên gồm tối đa 2n sensor. Sensor (`N`,v) và (`L`,v) là hai ứng viên khác nhau, có thể cùng được chọn và được tính là hai sensor. Không áp đặt ràng buộc chỉ được chọn một loại tại mỗi đỉnh trong mô hình mặc định.

**Quy ước riêng của script thực nghiệm theo yêu cầu cập nhật:** greedy vẫn xem (`N`,v) và (`L`,v) là hai ứng viên riêng và tính gain đúng như mô hình trên, nhưng cột `code_size` đếm số **vị trí đỉnh vật lý khác nhau**. Nếu cả hai loại được chọn tại cùng v thì vị trí đó chỉ đóng góp 1 vào `code_size`. Script ghi thêm `total_selected_types` (số ứng viên đã chọn) và `dual_type_vertex_count` để không trộn hai đại lượng. Đây là thay đổi ở metric báo cáo, không thay công thức chọn greedy.

**Bảo đảm nghiệm:** phải có sensor đơn đỉnh tại **mọi** v∈V, không có ngân sách hay ràng buộc khác ngăn chọn toàn bộ chúng. Khi chọn tất cả n sensor đơn đỉnh, vector tín hiệu chính là vector chỉ thị của F. Vì thế mọi tập cháy đều phân biệt được, và mọi tập cháy khác rỗng đều có tín hiệu khác 0. Lập luận này đúng cả với K lớn hơn 2.

Chỉ thêm sensor đơn đỉnh ở một vài đỉnh không tự động bảo đảm nghiệm.

Một nghiệm có n sensor luôn tồn tại, nên OPT ≤ n. Tuy nhiên greedy hỗn hợp có thể chọn hơn n sensor; cận n là cận của nghiệm toàn sensor đơn đỉnh, không phải cận cho mọi lần chạy greedy.

## 3. Vì sao cần sensor đơn đỉnh?

Với đồ thị ví dụ có các cạnh (1,2),(1,5),(2,3),(2,4),(3,4),(4,5):

- N[2]={1,2,3,4}.
- N[3]={2,3,4} ⊆ N[2].

Hai trạng thái {2} và {2,3} cho cùng tín hiệu ở mọi sensor lân cận đóng, vì N[2] ∪ N[3]=N[2]. Bài K=2 chỉ dùng loại này vô nghiệm.

Sensor (`L`,3) phân biệt ngay hai trạng thái trên: tín hiệu lần lượt là 0 và 1.

## 4. Trạng thái cháy, signature và ràng buộc

Đặt:

\[
\mathcal F=\{F\subseteq V:1\le |F|\le2\},\qquad
M=n+\binom n2.
\]

Trạng thái rỗng không nằm trong F; ta xử lý nó qua điều kiện domination.

Với danh sách sensor đã chọn C=(a₀,…,aₜ₋₁), signature của đỉnh v là vector:

\[
s_C(v)=(\mathbf1[v\in D(a_0)],\ldots,\mathbf1[v\in D(a_{t-1})]).
\]

Signature trạng thái cháy là:

\[
s_C(F)=\bigvee_{v\in F}s_C(v).
\]

Hai yêu cầu tương ứng với Set Cover:

- Domination: s_C(F) ≠ 0 với mọi F∈F.
- Separation: s_C(F) ≠ s_C(F') với mọi F≠F'.

Universe về mặt toán học có M+choose(M,2) phần tử, nhưng không dựng nó trong code.

## 5. Dữ liệu cần lưu

### 5.1. Partition đỉnh gốc

Lưu danh sách các nhóm không rỗng Qᵢ; trong mỗi nhóm, mọi đỉnh có cùng signature sᵢ:

```python
groups = [(signature_int, set_of_vertices), ...]
```

Các nhóm rời nhau, hợp bằng V, và có signature đôi một khác nhau. Tổng số phần tử được lưu trong mọi nhóm là n.

Dùng integer bitmask để lưu signature:

- Sensor được chọn thứ t dùng bit `1 << t`.
- OR signature bằng toán tử `|`.
- Signature rỗng là `0`.
- Không dùng chuỗi bit làm khóa trong bản triển khai chính.

**Phải lưu membership hoặc cấu trúc tương đương**, không chỉ lưu kích thước nhóm: còn cần biết đỉnh nào thuộc D(a) để tính gain và tách nhóm. Có thể thay set bằng danh sách đỉnh + `group_id[v]` hoặc bitset.

### 5.2. Bảng đếm trạng thái cháy theo signature

Lưu dictionary tạm/cached:

```python
counts[signature] = number_of_fire_states
```

Bảng này chỉ lưu số lượng, không lưu danh sách các tập F. Xây lại sau mỗi lần cập nhật partition.

Phân biệt rõ:

- Qᵢ: nhóm **đỉnh gốc**, có membership.
- `counts[s]`: số **trạng thái cháy** có signature s, chỉ là một số đếm.

Không được giả định rằng các Qᵢ đã là singleton thì bài K=2 đã giải xong. Signature của hai cặp đỉnh khác nhau vẫn có thể có cùng OR.

## 6. Hàm tổng hợp số lượng bằng tổ hợp

Với vector trọng số w=(w₁,…,wₚ), định nghĩa:

\[
T_s(w)=
\sum_{i:s_i=s}\left(w_i+\binom{w_i}{2}\right)
+
\sum_{i<j:s_i\lor s_j=s}w_iw_j.
\]

Ý nghĩa các hạng:

| Loại trạng thái | Signature | Số lượng |
|---|---|---:|
| Một đỉnh trong Qᵢ | sᵢ | wᵢ |
| Hai đỉnh khác nhau trong Qᵢ | sᵢ | choose(wᵢ,2) |
| Một đỉnh trong Qᵢ, một trong Qⱼ, i<j | sᵢ OR sⱼ | wᵢwⱼ |

Các nguồn tạo ra cùng signature phải được **cộng vào cùng khóa**. Không tính gain độc lập theo từng cặp nhóm: làm vậy sẽ bỏ sót separation giữa những trạng thái từ các nguồn khác nhau nhưng cùng OR.

Cho qᵢ=|Qᵢ|. Số trạng thái cháy có signature s là:

\[
m_s=T_s(q).
\]

Kiểm tra bất biến:

\[
\sum_s m_s=M.
\]

## 7. Gain của một sensor ứng viên

Với bất kỳ loại sensor a, đặt:

\[
h_i(a)=|Q_i\cap D(a)|,\qquad r_i(a)=q_i-h_i(a).
\]

rᵢ là số đỉnh trong Qᵢ không được a phát hiện. Các trạng thái không được a phát hiện phải chỉ gồm những đỉnh như vậy. Vì thế:

\[
z_s(a)=T_s(r),\qquad b_s(a)=m_s-z_s(a).
\]

- zₛ: số trạng thái signature s cho tín hiệu 0 tại a.
- bₛ: số trạng thái signature s cho tín hiệu 1 tại a.

Gain chính xác:

\[
\boxed{g_C(a)=b_0(a)+\sum_s b_s(a)z_s(a).}
\]

Giải thích:

1. b₀ là số trạng thái chưa được phát hiện nay được phát hiện.
2. Trong các trạng thái có cùng signature s, mỗi cặp gồm một trạng thái cho tín hiệu 1 và một trạng thái cho tín hiệu 0 được phân biệt lần đầu. Có bₛzₛ cặp; không chia 2.
3. Các trạng thái đã khác signature không thể trùng trở lại khi thêm bit, nên không được tính thêm vào gain.

Công thức chỉ dùng D(a), nên không cần hai phiên bản thuật toán cho hai loại sensor.

Với sensor đơn đỉnh (`L`,v), chỉ nhóm chứa v có hᵢ=1, các nhóm khác có hᵢ=0. Đây là cơ hội tối ưu sau; bản đầu cứ dùng hàm tổng quát để dễ kiểm chứng.

## 8. Cập nhật partition và điều kiện dừng

Sau khi chọn a ở vòng t, với từng nhóm (s,Q):

- Q⁻=Q\D(a), signature vẫn là s.
- Q⁺=Q∩D(a), signature mới là `s | (1 << t)`.
- Bỏ nhóm rỗng; tăng t sau khi cập nhật.

Các nhóm cũ không bao giờ gộp lại vì phần signature cũ của chúng đã khác nhau. Một nhóm không tách đôi vẫn có thể phải cập nhật signature nếu mọi đỉnh của nó đều được a phát hiện.

Số ràng buộc còn lại:

\[
R(C)=m_0+\sum_s\binom{m_s}{2}.
\]

Dừng thành công khi R(C)=0. Kiểm tra khi debug:

\[
R(C\cup\{a\})=R(C)-g_C(a).
\]

Nếu R>0 mà mọi ứng viên có gain 0 thì family hiện tại không đủ, hoặc code sai. Với đủ sensor đơn đỉnh và đúng mô hình, tình huống này không thể xảy ra: một trạng thái chưa detect chứa đỉnh v có sensor đơn đỉnh chưa chọn; hai trạng thái trùng signature nhưng khác nhau có v thuộc hiệu đối xứng, và sensor đơn đỉnh tại v chưa chọn sẽ phân biệt chúng.

Do đó greedy luôn tiến triển và kết thúc sau nhiều nhất số ứng viên (≤2n) vòng. Điều này không có nghĩa greedy tìm được số sensor nhỏ nhất.

## 9. Mã Python tham chiếu

Mã dưới đây không sinh danh sách trạng thái cháy hay Universe. `candidates` là mapping có thứ tự từ ID sensor đến tập phát hiện; ID có thể là tuple (`N`,v) hoặc (`L`,v). Tie-break theo thứ tự chèn ứng viên. Input phải có D(a)⊆V.

```python
from collections import defaultdict
from itertools import combinations

def choose2(t):
    return t * (t - 1) // 2

def aggregate(groups, weights):
    out = defaultdict(int)
    for i, (sig, _) in enumerate(groups):
        w = weights[i]
        out[sig] += w + choose2(w)
        for j in range(i):
            out[sig | groups[j][0]] += w * weights[j]
    return {s: count for s, count in out.items() if count}

def remaining(counts):
    return counts.get(0, 0) + sum(choose2(m) for m in counts.values())

def candidate_gain(groups, counts, detection):
    r = [len(vertices - detection) for _, vertices in groups]
    zeros = aggregate(groups, r)
    gain = counts.get(0, 0) - zeros.get(0, 0)
    for sig, m in counts.items():
        z = zeros.get(sig, 0)
        gain += (m - z) * z
    return gain

def refine(groups, detection, bit):
    result = []
    for sig, vertices in groups:
        hit = vertices & detection
        miss = vertices - detection
        if miss:
            result.append((sig, miss))
        if hit:
            result.append((sig | bit, hit))
    return result

def greedy_k2(vertices, candidates):
    # candidates: ordered mapping sensor_id -> set of detected vertices
    groups = [(0, set(vertices))] if vertices else []
    available = dict(candidates)
    selected, trace = [], []
    while True:
        counts = aggregate(groups, [len(q) for _, q in groups])
        before = remaining(counts)
        if before == 0:
            return selected, trace
        gains = {a: candidate_gain(groups, counts, d)
                 for a, d in available.items()}
        if not gains or max(gains.values()) == 0:
            raise ValueError('Infeasible candidate family, or implementation error')
        # Ties: first sensor in candidate insertion order.
        best = max(gains, key=gains.get)
        detection = available.pop(best)
        groups = refine(groups, detection, 1 << len(selected))
        selected.append(best)
        after_counts = aggregate(groups, [len(q) for _, q in groups])
        after = remaining(after_counts)
        assert after == before - gains[best]
        trace.append((best, gains[best], before, after))
```

## 10. Ví dụ chạy lại được

```python
V = set(range(1, 6))
N = {
    1: {1, 2, 5},
    2: {1, 2, 3, 4},
    3: {2, 3, 4},
    4: {2, 3, 4, 5},
    5: {1, 4, 5},
}
candidates = {('N', v): N[v] for v in sorted(V)}
candidates.update({('L', v): {v} for v in sorted(V)})
selected, trace = greedy_k2(V, candidates)
```

### 10.1. Vòng đầu thay đổi khi thêm loại sensor mới

Với d=|D(a)| và M=n+choose(n,2), số trạng thái được a detect là:

\[
H(a)=d+\binom d2+d(n-d).
\]

Gain ban đầu:

\[
g_\varnothing(a)=H(a)+H(a)(M-H(a)).
\]

Với n=5, M=15:

| Ứng viên | d | H | Gain |
|---|---:|---:|---:|
| (`N`,1), (`N`,3), (`N`,5) | 3 | 12 | 48 |
| (`N`,2), (`N`,4) | 4 | 14 | 28 |
| Mọi sensor (`L`,v) | 1 | 5 | 55 |

Vì vậy greedy hỗn hợp chọn sensor đơn đỉnh ngay vòng đầu. Không được giữ nguyên các bước chọn N1, N3, N5 của bài chỉ dùng sensor lân cận rồi gọi đó là cùng một lần chạy greedy hỗn hợp.

### 10.2. Trace cho cách tie-break trong code

| Vòng | Sensor chọn | Gain | R trước | R sau |
|---|---|---:|---:|---:|
| 1 | (`L`,1) | 55 | 120 | 65 |
| 2 | (`L`,2) | 32 | 65 | 33 |
| 3 | (`L`,3) | 18 | 33 | 15 |
| 4 | (`L`,4) | 10 | 15 | 5 |
| 5 | (`L`,5) | 5 | 5 | 0 |

Kết quả này minh họa greedy, không phải tuyên bố rằng 5 là tối ưu.

Sau vòng 1: partition đỉnh gốc gồm {1} có signature bitmask 1 và {2,3,4,5} có signature 0. Số trạng thái cháy tương ứng là m₁=5 và m₀=10. Khi xét (`L`,2), z₀=3+choose(3,2)=6, z₁=1+1·3=4. Vì thế gain = (10−6)+(10−6)·6+(5−4)·4 = 32.

### 10.3. Kiểm tra trường hợp suy biến

- n=0: không có trạng thái cháy; trả nghiệm rỗng.
- n=1: một sensor đơn đỉnh là đủ.
- Đồ thị đầy đủ: sensor lân cận đều phát hiện toàn V; sensor đơn đỉnh vẫn bảo đảm nghiệm.
- Đỉnh cô lập: hai loại sensor tại đỉnh đó có cùng D. Có thể gộp các ứng viên cùng D nếu chi phí bằng nhau và không có ràng buộc riêng theo loại; không bắt buộc.
- Hai đỉnh có cùng closed neighborhood: không được gộp/xóa chúng khỏi bài toán, vì chúng là hai nguồn cháy khác nhau và sensor đơn đỉnh có thể phân biệt chúng.

## 11. Độ phức tạp và giới hạn

Gọi p là số nhóm đỉnh gốc hiện tại, L là số signature trạng thái cháy có số lượng dương, A là số ứng viên còn lại.

- p≤n; L≤M=O(n²), đồng thời L≤2^|C|.
- Xây m từ các nhóm: O(p²) phép cộng/OR/tra dictionary.
- Với một ứng viên: tính r tốn O(n) bằng phép kiểm tra membership, rồi xây z và tính gain tốn O(p²+L). Vì L=O(p²), tổng là O(n+p²).
- Mỗi vòng: O(p²+A(n+p²)); với A≤2n là O(n(n+p²)).
- Cận thô xấu nhất toàn bộ greedy là O(n⁴) với ≤2n vòng; không phải cải tiến thành tuyến tính hay gần tuyến tính.
- Bộ nhớ partition là O(n) membership; bảng m,z tốn O(L) bản ghi. Không lưu đồng thời z cho mọi ứng viên; tính tuần tự và giữ ứng viên tốt nhất.
- Closed-neighborhood matrix nếu dùng tốn O(n²); có thể thay bằng adjacency list/bitset.

Các cận trên xem thao tác signature và số nguyên là đơn vị. Thực tế signature dài |C| bit, nên OR/hash và lưu khóa tăng theo độ dài bit. L có thể đạt Θ(n²) khi gần nghiệm: không có cam kết luôn lưu ít hơn bậc hai các số đếm.

Với K=2 cố định, số trạng thái tăng O(n²), Universe explicit tăng O(n⁴); đây là đa thức theo n. Công thức `aggregate` trong note chỉ dành cho K=2. Mô hình sensor bảo đảm nghiệm với mọi K, nhưng không được dùng nguyên hàm đếm này cho K>2.

## 12. Nếu muốn ưu tiên sensor lân cận

Mặc định chọn gain lớn nhất trên cả hai loại với chi phí bằng nhau. Nếu mục tiêu thực tế khác, phải chọn rõ một biến thể:

1. **Có chi phí:** đặt cost(a)>0, chọn gain(a)/cost(a). Đây là greedy cho weighted Set Cover; mục tiêu là tổng chi phí, không còn chỉ là số sensor. Có thể đặt chi phí sensor đơn đỉnh cao hơn nếu có lý do thực tế.
2. **Bổ sung khi bị kẹt:** ban đầu chỉ cho phép sensor lân cận; khi còn ràng buộc nhưng mọi sensor lân cận có gain 0, mở sensor đơn đỉnh. Vẫn bảo đảm kết thúc nếu đủ sensor đơn đỉnh, nhưng không phải greedy trên toàn family từ đầu và có thể chọn nhiều sensor dư thừa.

Không tự ý áp dụng biến thể thứ hai chỉ vì gọi sensor đơn đỉnh là phương án cứu nghiệm. Bản code trong note dùng mặc định thứ nhất của mô hình: hai loại tham gia đồng thời, chi phí bằng nhau.

Nếu chỉ tối thiểu số sensor, có thể giữ sẵn nghiệm dự phòng gồm n sensor đơn đỉnh và trả nghiệm tốt hơn giữa nó và kết quả greedy. Đây là bước hậu xử lý tùy chọn, không có trong code tham chiếu.

## 13. Checklist triển khai và kiểm chứng

- [ ] ID phân biệt loại sensor và vị trí.
- [ ] N[v] chứa chính v; đồ thị vô hướng và membership nhất quán.
- [ ] Có đủ (`L`,v) cho mọi v nếu yêu cầu bảo đảm nghiệm.
- [ ] `choose2(t)=t*(t-1)//2`; không dùng số thực.
- [ ] Signature dùng cùng thứ tự bit cho mọi nhóm; bit mới không được tái sử dụng.
- [ ] Cộng chung mọi đóng góp có cùng OR signature.
- [ ] Không chỉ lưu kích thước Q: cần membership để tách nhóm.
- [ ] Không dừng chỉ vì mọi nhóm đỉnh gốc là singleton.
- [ ] Dừng bằng R=0; kiểm tra R_new=R_old−gain.
- [ ] Tổng m bằng n+choose(n,2), và 0≤z_s≤m_s.
- [ ] Sensor đã chọn bị loại khỏi danh sách ứng viên.
- [ ] Dùng số nguyên đủ lớn: R có thể đạt Θ(n⁴). Python int phù hợp; C++ cần kiểm tra giới hạn và phép nhân trung gian.
- [ ] Chỉ ở test nhỏ: liệt kê F và so sánh gain với đếm trực tiếp domination/separation.

Mã tham chiếu đã được đối chiếu với cách đếm trực tiếp trên 84 đồ thị ngẫu nhiên kích thước n=1,…,7 (seed 20260922), gồm 6.720 phép so sánh gain qua các prefix sensor khác nhau, và kiểm tra signature nghiệm cuối. Đây là kiểm tra tính đúng của triển khai nhỏ, không phải benchmark hiệu năng trên đồ thị lớn.

## 14. Tóm tắt cấu trúc để bắt đầu code

```text
Input: V, closed neighborhoods, candidate sensor detection sets
Persistent state: selected sensor IDs, original-vertex groups with signatures
Per iteration: counts = aggregate(group sizes)
Per candidate: zeros = aggregate(group sizes minus detected vertices)
Gain: counts[0] - zeros[0] + sum((counts[s]-zeros[s])*zeros[s])
Select: maximum gain (stable tie-break)
Update: split each original-vertex group and append a signature bit
Stop: counts[0] + sum(choose2(counts[s])) == 0
```

Không cần materialize các trạng thái cháy trong production. Không cần dựng các dòng XOR của ma trận ràng buộc. Phần chung giữa hai loại sensor được biểu diễn hoàn toàn bằng tập D(a).
