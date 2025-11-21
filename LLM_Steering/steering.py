def get_steering_hook(steering_vec, steering_coeff):
    def steering_hook(module, input, output):
        if isinstance(output,tuple):
            output[0][0,-1,:]+=steering_coeff*steering_vec
        else:
            output[0,-1,:]+=steering_coeff*steering_vec
        return output
    return steering_hook
