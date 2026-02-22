# Define function for data masks
def mask_pos_slice(pos_i, tar_pos, thickness = 0.005):
    import numpy as np
    # Plot at a specific z position
    mask = np.abs(pos_i - tar_pos) < thickness
    return mask

def mask_pos_range(pos_i, pos_start, pos_end):
    # Plot a sector of a ring at a z position
    mask_pos = (pos_i > pos_start) & (pos_i < pos_end)
    return mask_pos

def mask_line(gas_pos, x_pos = None, y_pos = None, z_pos = None, thickness = 0.005, plot_range = 1.):
    import numpy as np
    # Plot the line along a specific direction given a position at the plane of the rest two axes
    if x_pos is None:
        mask = (np.abs(gas_pos[:, 1] - y_pos) <= thickness) & (np.abs(gas_pos[:, 2] - z_pos) <= thickness) #&\
               #(np.abs(gas_pos[:, 0] - x_pos) <= plot_range)
    elif y_pos is None:
        mask = (np.abs(gas_pos[:, 0] - x_pos) <= thickness) & (np.abs(gas_pos[:, 2] - z_pos) <= thickness) #&\
               #(np.abs(gas_pos[:, 1] - y_pos) <= plot_range)
    elif z_pos is None:
        mask = (np.abs(gas_pos[:, 0] - x_pos) <= thickness) & (np.abs(gas_pos[:, 1] - y_pos) <= thickness) #&\
               #(np.abs(gas_pos[:, 2] - z_pos) <= plot_range)
    else:
        raise ValueError("`Two of x_pos, y_pos, or z_pos must be specified")
    return mask

# Define function for plots
def plot_gas_2d(gas_pos_masked, masked_quantities, plot_axes = [0, 2], 
                cmaps = None, vmin = None, vmax = None, marker_size = 0.005, 
                sharex=True, sharey=True):
    n_plots = len(masked_quantities)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, n_plots, figsize = (6*n_plots, 6), dpi = 150)
    for i in range(n_plots):
        quantity = masked_quantities[i]
        scatter_kwargs = {
            'c': quantity,
            's': marker_size
        }
        if cmaps is not None and i < len(cmaps):
            scatter_kwargs['cmap'] = cmaps[i]
        if vmin is not None and i < len(vmin):
            scatter_kwargs['vmin'] = vmin[i]
        if vmax is not None and i < len(vmax):
            scatter_kwargs['vmax'] = vmax[i]
        ax[i].scatter(gas_pos_masked[:, plot_axes[0]], gas_pos_masked[:, plot_axes[1]], 
                      **scatter_kwargs)
        ax[i].set_aspect('equal', adjustable='box')
    plt.tight_layout()
    return fig, ax
    
def plot_gas_2d_subplots(gas_pos, masks=[], planes=[], quantities=[], plot_w_size=3, plot_h_size=6,
                        cmaps=None, cbars=None, marker_size=0.005, titles=None, 
                        cbar_labels=None, vmin=None, vmax=None, sharex=True, sharey=True):
    import matplotlib.pyplot as plt
    n_plots = len(quantities)
    
    if not (len(masks) == len(planes) == len(quantities)):
        raise ValueError(f"masks ({len(masks)}), planes ({len(planes)}), and quantities ({len(quantities)}) must have the same length")
    fig, ax = plt.subplots(1, n_plots, figsize=(plot_w_size*n_plots, plot_h_size), 
                           dpi=200, sharex=sharex, sharey=sharey)
    if n_plots == 1:
        ax = [ax]
    
    plane_config = {'xy': (0, 1, 'X [kpc]', 'Y [kpc]'),
                    'yz': (1, 2, 'Y [kpc]', 'Z [kpc]'),
                    'xz': (0, 2, 'X [kpc]', 'Z [kpc]')}
    
    for i in range(n_plots):
        quantity = quantities[i]
        plane = planes[i].lower()  # Handle case-insensitive input
        mask = masks[i]

        if plane not in plane_config:
            raise ValueError(f"Invalid plane '{plane}'. Must be one of: xy, yz, xz")
        
        ax_ind0, ax_ind1, xlabel, ylabel = plane_config[plane]
        
        scatter_kwargs = {
            'c': quantity[mask],
            's': marker_size
        }
        if cmaps is not None and i < len(cmaps):
            scatter_kwargs['cmap'] = cmaps[i]
        if vmin is not None and i < len(vmin):
            scatter_kwargs['vmin'] = vmin[i]
        if vmax is not None and i < len(vmax):
            scatter_kwargs['vmax'] = vmax[i]
            
        im = ax[i].scatter(gas_pos[mask, ax_ind0], gas_pos[mask, ax_ind1], **scatter_kwargs)
        
        if cbars is not None and i < len(cbars) and cbars[i]:
            cbar = fig.colorbar(im, ax=ax[i])
            if cbar_labels is not None and i < len(cbar_labels):
                cbar.set_label(cbar_labels[i])
        
        if titles is not None and i < len(titles):
            ax[i].set_title(titles[i])
        
        ax[i].set_xlabel(xlabel)
        ax[i].set_ylabel(ylabel)
        ax[i].set_aspect('equal', adjustable='box')
    
    all_same_plane = len(set(planes)) == 1
    if all_same_plane:
        plane = planes[0].lower()
        _, _, xlabel, ylabel = plane_config[plane]
        for a in ax:
            a.set_xlabel('')
            a.set_ylabel('')
        fig.supxlabel(xlabel)
        fig.supylabel(ylabel)
    plt.tight_layout()
    return fig, ax

def plot_gas_line_profiles(gas_pos, masks=[], axis=[], quantities=[], color_quantities=[], ylabel=[], 
                      plot_w_size=8, plot_h_size=2.5, cmaps=None, marker_size=0.005, 
                      titles=None, vmin=None, vmax=None, cbar_labels=None):
    import matplotlib.pyplot as plt
    import numpy as np
    n_plots = len(quantities)
    
    if not (len(masks) == len(axis) == len(quantities)):
        raise ValueError(f"masks ({len(masks)}), axis ({len(axis)}), and quantities ({len(quantities)}) must have the same length")
    
    fig, ax = plt.subplots(n_plots, 1, figsize=(plot_w_size, plot_h_size*n_plots), 
                           dpi=100, sharex=True, sharey=False)
    if n_plots == 1:
        ax = [ax]
        
    axis_config = {
        'x': (0, 'X [kpc]'),
        'y': (1, 'Y [kpc]'),
        'z': (2, 'Z [kpc]')
    }
    
    for i in range(n_plots):
        quantity = quantities[i]
        dirc = axis[i].lower()
        mask = masks[i]
        
        # Validate axis
        if dirc not in axis_config:
            raise ValueError(f"Invalid axis '{dirc}'. Must be one of: x, y, z")
        ax_ind, xlabel = axis_config[dirc]
        
        # Handle color
        if color_quantities is not None and i < len(color_quantities):
            color = color_quantities[i][mask]
        else:
            color = 'purple'
        
        # Build scatter kwargs
        scatter_kwargs = {
            'c': color,
            's': marker_size
        }
        if cmaps is not None and i < len(cmaps):
            scatter_kwargs['cmap'] = cmaps[i]
        if vmin is not None and i < len(vmin):
            scatter_kwargs['vmin'] = vmin[i]
        if vmax is not None and i < len(vmax):
            scatter_kwargs['vmax'] = vmax[i]
        
        im = ax[i].scatter(gas_pos[mask, ax_ind], np.log10(quantity[mask]), **scatter_kwargs)
        if color_quantities is not None and i < len(color_quantities):
            cbar = fig.colorbar(im, ax=ax[i])
            if cbar_labels is not None and i < len(cbar_labels):
                cbar.set_label(cbar_labels[i])
        
        ax[i].set_xlabel(xlabel)
        if titles is not None and i < len(titles):
            ax[i].set_title(titles[i])
    
        if ylabel is not None:
            ax[i].set_ylabel(ylabel[i])
        else:
            ax[i].set_ylabel('log Quantity')
    plt.tight_layout()
    return fig, ax


def plot_gas_phase_space(quantity_x, quantity_y, mask=None, labels=None, bin_number=500):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    
    if mask is not None:
        h = ax.hist2d(quantity_x[mask], quantity_y[mask],
                      bins=bin_number, norm=LogNorm(),
                      cmap='viridis')
    else:
        h = ax.hist2d(quantity_x, quantity_y,
                      bins=bin_number, norm=LogNorm(),
                      cmap='viridis')
    ax[i].set_aspect('equal', adjustable='box')
    plt.colorbar(h[3], ax=ax)
    if labels is not None and len(labels) >= 2:
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
    return fig, ax