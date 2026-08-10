# Identifying Code và Set Cover Greedy (without Universe)
File này cũng trình bày về cách giải bài toán Identifying Code thông qua bài toán set cover (tham khảo file set-cover/greedy-set-cover.md để thêm thông tin). Tuy nhiên sự khác nhau là ta sẽ không build universe để tiết kiệm thời gian.


Bây giờ ta sẽ chạy thuật toán thông qua 1 ví dụ:
Cho graph 5 đỉnh và các cạnh (1,2), (1,5), (2,3), (2,4), (3,4), (4,5):

'''
1----2
|    | \  
|    |   \
5----4----3
'''

Đầu tiên ta sẽ xây dựng ma trận closed_neighborhood (đặt các đỉnh tương ứng là v_1, v_2, v_3, v_4, v_5).

      v_1 v_2 v_3 v_4 v_5
v_1 |  1   1   0   0   1 |
v_2 |  1   1   1   1   0 |
v_3 |  0   1   1   1   0 |
v_4 |  0   1   1   1   1 |
v_5 |  1   0   0   1   1 |

Bắt đầu chạy thuật toán:

GLOBAL variable:
TOTAL_CONSTRAIN = 5 + 2C5 = 15 (5 là điều kiện phủ, 2C5=10 là điều kiện phân biệt từng cặp).
VECTOR_1 = []
VECTOR_0 = []

Bước 1: chọn cột nào có thể phủ được nhiểu constrain nhất:

Candidate_v_1 = 2C5 - 3C2 - 2C2 + 3 = 10 - 3 - 1 + 3 = 9
giải thích: 
- 2C5: tổng constraint của từng cặp.
- -3C2: vì trong cột v_1 có 3 số 1 nên không thể phủ 3 cặp phân biệt trong số 1 này (lưu ý chỉ có thể là 2 số khác nhau thì mới phủ được).
- -2C2: tương tự nhưng với số 0 trong cột v_1.
- +3: vì có 3 số 1 nên phủ được điều kiện phủ.

Tương tự:
candidate_v_2 = 5C2 - 4C2 - 1C2 +4 = 8
candidate_v_3 = 2C5 - 3C2 - 2C2 + 3 = 9
candidate_v_4 =  5C2 - 4C2 - 1C2 +4 = 8
candidate_v_5 =  2C5 - 3C2 - 2C2 + 3 = 9

==> Chọn v_1 (vì có số constraint phủ lớn nhất = 9)

Updated GLOBAL variable:
TOTAL_CONSTRAIN = 15 - 9 = 6
VECTOR_1 = [v_1, v_2, v_5] (các ô số 1 của cột v_1)
VECTOR_0 = [v_3, v_4] (các ô số 0 của cột v_1)
lưu ý: VECTOR_1, VECTOR_0 ở đây để lưu những constrain chưa phủ.

Bước 2: đã chọn v_1, trong 4 candidate còn lại chọn cột nào phủ được nhiều constraint nhất.

Candidate_v_2 = 2  + 2 = 4
Giải thích:
- số 2 đầu tiên: vì v1 chưa phủ được v_3, v_4 (trong điều kiện phủ) mà cột v_2 phủ được v_3, v_4. Chỗ này mình có thể dùng phép OR (v1 OR v2) - v1 = 2
- số 2 tiếp theo:
  + Xử lý những constraint chưa được phủ trong VECTOR_1:
    (cột v_1 AND cột v_2) = (1,1,0,0,1) AND (1,1,1,1,0) = (1, 1, 0, 0, 0)
    DIFF = TỔng số 1 (cột v_1) - tổng số 1 ((cột v_1 AND cột v_2) ) = 3 - 2 = 1 (v_5 chỗ này v_5 của cột v_2 = 0, nên v_5 sẽ kết hợp được với v_1, v_2 giảm 2)
    2 = DIFF * tổng số 1 ((cột v_1 AND cột v_2) )
  + Xử lý những constraint chưa được phủ trong VECTOR_0:
    Tương tự nhưng dùng phép OR.

Candidate_v_3 = 2 + 2 = 4
Candidate_v_4 = 2 + 2 = 4
Candidate_v_5 = 1 + 3 = 4
Giải thích candidate_v_5.
- Ở đây lưu ý là v_5 chỉ phủ được v_4 (không phủ v_3) nên + 1.
- số 3 ở đây:
  + Xử lý constraint trong VECTOR_1:
    B = cột v_1 AND cột v_5 = (1,1,0,0,1) AND (1,0,0,1,1) = (1,0,0,0,1) (v_2 đã đổi sang 0, nên v_2 sẽ kết hợp được với v_1, v_5 trong VECTOR_1 để phủ các constrain cặp). ta phủ được 2 cặp (v_2, v_1), (v_2,v_5).
  + Xử lý constraint trong VECTOR_0:
    B = cột v_1 OR cột v_5 = (1,1,0,0,1) OR (1,0,0,1,1) = (1,1,0,1,1) (v_4 đã đổi). v_4 có thể kết hợp với v_3 để phủ constraint (v_4, v_3)

=> Giả sử mình sẽ chọn v_5 (vì candidate_v_5 = 4)

Updated GLOBAL variable:
TOTAL_CONSTRAIN = 6 - 4 = 2
VECTOR_1 = [v_1, v_5] (v_2 được loại khỏi VECTOR_1 vì cột v_5 đã làm điều đó)
VECTOR_0 = [v_3] (tương tự bỏ v_4 khỏi VECTOR_0)
lưu ý: VECTOR_1, VECTOR_0 ở đây để lưu những constraint chưa phủ.

..... tiếp tục cho đến khi TOTAL_CONSTRAIN = 0.....
