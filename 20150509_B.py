N = int(input())
S = str(input())
flag = 0
#手順の回数n
n = (N-1)//2
#最後の手順
f = n%3
while len(S) > 1 and flag == 0:
    if f == 0:
        if S[0] == 'b' and S[-1] == 'b':
            S = S[1:-1]
        else:
            flag = 1
    elif f == 1:
        if S[0] == 'a' and S[-1] == 'c':
            S = S[1:-1]
        else:
            flag = 1
    elif f == 2:
        if S[0] == 'c' and S[-1] == 'a':
            S = S[1:-1]
        else:
            flag = 1
if flag == 0:
    if S == 'b':
        print(n)
    else:
        print(-1)
else:
    print(-1)