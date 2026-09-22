from tensorboardX import SummaryWriter

writer=SummaryWriter("logs")

#writer.add_image()
for i in range(100):
    writer.add_scalar("y=9x",9*i,i)

writer.close()