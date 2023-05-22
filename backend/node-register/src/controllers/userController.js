const { User } = require('../models/user');
const userRepository = require('../repositories/userRepository');
const jwt = require('jsonwebtoken');
const config = require('../config');

async function create(req, res) {
  const { login, senha } = req.body;
  try {
    const userExists = await userRepository.findByLogin(login);

    if (userExists) {
      return res.status(409).json({ message: 'User already exists' });
    }

    const user = await userRepository.create({ login, senha });

    const token = jwt.sign({ id: user.id }, config.jwtSecret);

    return res.status(201).json({ token });
  } catch (error) {
    console.error(error);
    return res.status(500).json({ message: 'Internal Server Error' });
  }
}

async function update(req, res) {
  const { id } = req.params;
  const { senha } = req.body;

  try {
    const userExists = await userRepository.findById(id);

    if (!userExists) {
      return res.status(404).json({ message: 'User not found' });
    }

    await userRepository.update(id, senha);

    return res.sendStatus(204);
  } catch (error) {
    console.error(error);
    return res.status(500).json({ message: 'Internal Server Error' });
  }
}

async function findById(req, res) {
  const { id } = req.params;

  try {
    const user = await userRepository.findById(id);

    if (!user) {
      return res.status(404).json({ message: 'User not found' });
    }

    return res.json({ id: user.id, login: user.login, disabilitado: user.disabilitado });
  } catch (error) {
    console.error(error);
    return res.status(500).json({ message: 'Internal Server Error' });
  }
}

async function findAll(req, res){
    try {
        const user = await userRepository.findAll();
    
        if (!user || users.length === 0) {
          return res.status(404).json({ message: 'User not found' });
        }
    
        const users = users.map(user => ({
            id: user.id,
            login: user.login,
            disabilitado: user.disabilitado
          }));
        
        return res.json(users)
      } catch (error) {
        console.error(error);
        return res.status(500).json({ message: 'Internal Server Error' });
      }
}

module.exports = {
  create,
  update,
  findById,
  findAll
};
