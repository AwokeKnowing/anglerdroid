def drepr(self):
    attr = [f'{k}="{v}"' if isinstance(v,str) else f'{k}={v}' 
            for k, v in self.__dict__.items()]
    print( f'{self.__class__.__name__}({", ".join(attr)})')

class Motion2D:
    def __init__(self, **kwargs):
        self.tf = None
        self.tf_inv = None
        self.theta = None
        self.tx = None
        self.ty = None
        self.lin_vel = None
        self.ang_vel = None
        self.lin_dist = None
        self.arc_dist = None
        self.arc_rad = None
        self.seconds = None
        
        self.__dict__.update(kwargs)




