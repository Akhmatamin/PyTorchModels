import random as rnd
print('')
print('🚩Hello!, here you should guess the random number in range 1 to 100🚩')
print('⚠️Important! You have 10 attempts.⚠️')
print('Good luck!💪')
print('')


random_num = rnd.randint(0, 100)
# print(random_num)

attempts = 10
while attempts > 0:
    user_num = int(input('Number is: '))
    if user_num == random_num:
        print('🏆You got it! Congrats🎉')
        print('✅Random number is:', random_num)
        if attempts == 10:
            print('You got it in first try!')
        else:
            print(f'You got it in {10 - attempts} try!')
        break
    elif user_num > random_num:
        print('❌Oops, it was high try again!')
        attempts -= 1
        print('❗Attempts left:', attempts)
    elif user_num < random_num:
        print('❌Oops, it was low try again!')
        attempts -= 1
        print('❗Attempts left:', attempts)