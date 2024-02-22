



def get_next_factor(dl, random_state, f, index=0, fov_to_ignore=None, values_to_ignore=None ):
    """Given FoV f, return the sample with k different FoV."""

    z = f.copy()

    # sample next factor
    new = random_state.choice(range(dl.factors_sizes[index]))

    z[:, index] = new

    return z


def get_next_different_factor(dl, random_state, f, index=0,  fov_to_ignore=None, values_to_ignore=None):
    """Given FoV f, return the sample with k different FoV."""

    z = f.copy()

    for i in range(z.shape[0]):

        # sample next factor
        new = random_state.choice(range(dl.factors_sizes[index]))

        # ensure it is different
        old = z[i, index]
        z[i, index] = new
        while old == new and not fov_is_valid(z[i], fov_to_ignore, values_to_ignore):
            new = random_state.choice(range(dl.factors_sizes[index]))
            z[i, index] = new

    return z


def fov_is_valid(f, fov_to_ignore=None, values_to_ignore=None):
    """Return true if factor does not contain values to ignore"""

    if fov_to_ignore is None:
        return True

    if isinstance(values_to_ignore, int):
        values_to_ignore = [values_to_ignore]

    for fov, values in zip(fov_to_ignore, values_to_ignore):

        if f[fov] in values:
            return False

    return True


