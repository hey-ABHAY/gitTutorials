import random
from itertools import count

print("<---WELCOME--->")
print("SNAKE--WATER--GUN")

n=int(input("ENTER NUMBER OF ROUNDS:"))

options=['s','w','g']

round=1
user_win=0
comp_win=0

while round<=n:
    print(f"Round{round}:\nSnake-'s'\nWater-'w'\nGun-'g'")
    try:
        player=input("choose your option:")
    except EOFError as e:
        print(e)
    if player!='s'and player!='w'and player!='g':
        print("Invalid entry , try AGAIN:")
        continue

    computer=random.choice(options)

    if computer=='s':
        if player=='w':
            comp_win+=1
        elif  player=='g':
            user_win+=1
    elif computer=='w':
        if player=='s':
            user_win+=1
        elif player=='g':
            comp_win+=1
    else:
        if player=='w':
            user_win+=1
        elif player=='s':
            comp_win+=1

    if(user_win>comp_win):
        print(f"You won round {round}\n")
    elif(comp_win>user_win):
        print(f"Computer won the round {round}\n")
    else:
        print(f"Round {round} draw")
    round+=1


if(user_win>comp_win):
    print("Congratulation you won the tournament!!!")
elif(comp_win>user_win):
    print("OOOPS,You loose! Better luck next time")
else:
    print("MATCH DRAW")