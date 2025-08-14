from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('logs')

# writer.add_image('test', img_tensor, 0)

#y=x
for i in range(10):
    writer.add_scalar('y=x', i, i)

#y=x^2
for i in range(10):
    writer.add_scalar('y=x^2', i*i, i)


writer.close()