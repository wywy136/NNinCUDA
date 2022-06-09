from pprint import pprint
import matplotlib.pyplot as plt


loss = [6.738090515136719, 1.4952598810195923, 0.9371780157089233, 0.9016490578651428, 0.9033308625221252, 0.9111259579658508, 0.9185771346092224, 0.9252322316169739, 0.9304689764976501, 0.9355436563491821]

x = [i for i in range(10)]
x_label = [500*i + 1 for i in x]

f = plt.figure()
f.set_figwidth(10)
f.set_figheight(6)

plt.xticks(x, x_label, rotation=50)

loss_epoch, = plt.plot(x, loss, c='b')

plt.xlabel("Training epoch")
plt.ylabel("Cost/Loss")
plt.savefig("./loss.png")